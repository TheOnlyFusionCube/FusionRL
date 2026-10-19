"""End-to-end CPU execution of FusionRL's actual learner and control plane.

The synthetic symbol environment is intentionally cheap and fully inspectable.
Async mode simulates delayed completions; it does not claim GPU overlap.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .config import FusionConfig
from .learner import learner_loss
from .pipeline import GroupSupervision,prepare_step
from .rewards import deepseek_effort_penalties, mimo_length_penalties
from .scheduler import FusionScheduler, GroupSpec, SchedulerConfig
from .trajectory import Assessment, ExecutionRecord, GenerationEvent, RolloutGroup, TrajectoryPath


def conditioning_key(payload: dict[str,Any]) -> str:
    """Hash task, environment, effort, harness and grader CONFIG, not its outcome."""
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def run_demo(config: FusionConfig, output: str | Path, *, progress: bool = True) -> dict[str,Any]:
    import torch
    from .tiny_policy import TinyCausalPolicy

    config.validate()
    if config.policy.model_kind != "dense":
        raise ValueError("The CPU model is dense; MoE execution requires an external replay-capable backend")
    if not set(config.reward.effort_levels) <= {25,50,75,100}:
        raise ValueError("The synthetic demo encodes efforts 25,50,75,100; real backends use the scalar prompt")
    if config.train.steps % config.schedule.burst_steps:
        raise ValueError("Demo must end at a publication boundary: steps must be a multiple of burst_steps")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("summary.json","metrics.jsonl","checkpoint.pt")):
        raise FileExistsError("Choose a fresh output directory; existing run evidence is not overwritten")
    torch.set_num_threads(1)
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    rng=random.Random(config.train.seed)
    generator=torch.Generator(device="cpu").manual_seed(config.train.seed+1)
    model=TinyCausalPolicy()
    model.eval()  # Dropout stays disabled while gradients remain enabled.
    optimizer=torch.optim.Adam(model.parameters(),lr=config.train.learning_rate)
    starting=model.evaluate(config.reward.effort_levels,config.train.max_generated_tokens)
    prompts=[model.prompt(symbol,b) for symbol in range(8) for b in config.reward.effort_levels]
    labels=torch.tensor([symbol for symbol in range(8) for _ in config.reward.effort_levels])
    for _ in range(config.train.sft_steps):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model.logits(prompts),labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.train.gradient_clip)
        optimizer.step()
    after_sft=model.evaluate(config.reward.effort_levels,config.train.max_generated_tokens)
    # A new optimizer explicitly separates SFT momentum from RL.
    optimizer=torch.optim.Adam(model.parameters(),lr=config.train.learning_rate)
    scheduler_options=asdict(config.schedule)
    scheduler_options.pop("dispatch_policy")
    scheduler_options.pop("sample_reuse")
    schedule=SchedulerConfig(**scheduler_options)
    controller=FusionScheduler(schedule)
    task_meta:dict[str,dict[str,Any]]={}
    groups:dict[str,RolloutGroup]={}
    trace:dict[str,Any]|None=None
    group_counter=0

    def refill() -> None:
        nonlocal group_counter,trace
        specs=[]
        missing=schedule.target_depth-controller.pending_count
        for _ in range(missing):
            group_id=f"g{group_counter:06d}"
            group_counter+=1
            symbol=rng.randrange(8)
            effort=rng.choice(config.reward.effort_levels)
            meta={"task":"synthetic-symbol-copy-v1","symbol":symbol,"effort":effort,
                  "harness":"in-process-pure-symbol-v1","environment":"symbol-v1",
                  "rubric":"format-v1","reward_route":config.reward.route,"config_epoch":0}
            task_meta[group_id]=meta
            specs.append(GroupSpec(group_id,conditioning_key(meta)))
        tickets=controller.dispatch(specs)
        completed=[]
        for ticket in tickets:
            meta=task_meta[ticket.group_id]
            prompt=model.prompt(meta["symbol"],meta["effort"])
            executions=[]
            for replicate in range(ticket.expected_executions):
                execution_id=f"{ticket.group_id}:e{replicate}"
                tokens,logps=model.sample(prompt,max_tokens=config.train.max_generated_tokens,generator=generator)
                outcome=float(tokens[-1]==meta["symbol"])
                event=GenerationEvent(execution_id+":request0",prompt,tokens,logps,
                    tuple(ticket.dispatch_version for _ in tokens),tuple("policy" for _ in tokens))
                path=TrajectoryPath(execution_id+":main",(event.event_id,))
                state_id=hashlib.sha256(json.dumps([prompt,tokens]).encode()).hexdigest()
                executions.append(ExecutionRecord(execution_id,ticket.group_id,(event,),(path,),
                    Assessment(outcome,True,state_id),conditioning_key=ticket.conditioning_key))
            group=RolloutGroup(ticket.group_id,tuple(executions),ticket.expected_executions,
                               ticket.dispatch_version,ticket.conditioning_key)
            groups[group.group_id]=group
            completed.append(group)
            if trace is None:
                trace={"schema":"fusionrl-rollout-group-v1","task":meta,"group":asdict(group)}
        # Long groups intentionally report after short groups. The controller's
        # preassigned membership must stay unchanged by this completion order.
        completed.sort(key=lambda g:sum(len(e.events[0].output_token_ids) for e in g.executions))
        for group in completed:
            controller.complete(group)

    started=time.perf_counter()
    refill()
    metrics=[]
    for step in range(config.train.steps):
        admitted=controller.begin_update()
        if admitted is None:
            raise RuntimeError("Synthetic executor completed all groups but the controller cannot form a quota batch")
        supervisions={}
        for group in admitted:
            original=groups[group.group_id]
            outputs=[e.events[0].output_token_ids for e in original.executions]
            reasoning=np.array([sum(t>=8 for t in tokens) for tokens in outputs],dtype=float)
            lengths=np.array(list(map(len,outputs)),dtype=float)
            costs=None
            r=config.reward
            if r.length_mode=="effort":
                costs=deepseek_effort_penalties(reasoning,budget=task_meta[group.group_id]["effort"],
                    min_budget=min(r.effort_levels),k0=r.effort_k0,tau=r.effort_tau,
                    length_norm=r.effort_length_norm,cap=r.effort_cap)
            elif r.length_mode=="group_relative":
                costs=mimo_length_penalties(lengths,group.rewards,min_pass_rate=r.length_min_pass_rate,
                    percentile=r.length_percentile,max_deduction=r.length_max_deduction,
                    tolerance=r.length_tolerance,saturation=r.length_saturation,exponent=r.length_exponent)
            # A deterministic toy rubric measures invalid structural symbols,
            # not length: symbol 9 is a permitted reasoning marker, 8/10..13 are
            # malformed format output. This does not implement MiMo's private
            # agentic grader; real runners must supply independently audited scores.
            quality=[1.0 if all(t<8 or t==9 for t in tokens) else .5 for tokens in outputs]
            kw:dict[str,Any]={}
            if r.route=="grs":
                kw={"solution_scores":[1.]*len(outputs),"behavior_scores":quality}
            elif r.route=="gar":
                kw={"quality_factors":quality}
            score_override=tuple(np.asarray(group.rewards)*np.asarray(quality)) if r.route=="scalar" else None
            supervisions[group.group_id]=GroupSupervision(
                execution_ids=tuple(e.execution_id for e in group.executions),
                artifact_ids=tuple(e.assessment.artifact_id for e in group.executions),
                conditioning_key=group.conditioning_key,route=r.route,scores=score_override,
                cost_penalties=tuple(costs) if costs is not None else None,gar_cap=r.gar_cap,**kw)
        prepared=prepare_step(admitted,supervisions)
        batch=prepared.batch
        zero_signal=sum(not result.diagnostics["has_learning_signal"] for result in prepared.grading.values())
        optimizer.zero_grad(set_to_none=True)
        loss,stat=learner_loss(batch,model.log_probs,config=config.loss)
        optimizer_applied=any(bool(targets) and abs(advantage)>1e-12 for targets,advantage in zip(batch.targets,batch.advantages))
        if optimizer_applied:
            loss.backward()
            gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.train.gradient_clip).item())
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite policy gradient; update not committed")
            # A wholly clipped batch must not move through stale Adam momentum.
            optimizer_applied=gradient_norm>0
            if optimizer_applied:
                optimizer.step()
        else:
            gradient_norm=0.0
        token_lags=[controller.policy_version-t.policy_version for g in admitted for e in g.executions for t in e.targets]
        stat.update({"step":step+1,"behavior_lag_max":max(token_lags,default=0),
            "behavior_lag_mean":float(np.mean(token_lags)) if token_lags else 0.,
            "reward_mean":float(np.mean([e.reward for g in admitted for e in g.executions])),
            "zero_signal_groups":zero_signal,"optimizer_applied":optimizer_applied,
            "gradient_norm":gradient_norm,"masked_stale":sum(e.masked_stale for g in admitted for e in g.executions)})
        controller.commit_update()
        for group in admitted:
            del groups[group.group_id]
            del task_meta[group.group_id]
        if controller.needs_publication:
            controller.publish()
            if step+1 < config.train.steps:
                refill()
        metrics.append(stat)
        if progress and ((step+1)%20==0 or step+1==config.train.steps):
            print(f"step {step+1}/{config.train.steps}: reward={stat['reward_mean']:.3f}, loss={stat['loss']:.4f}, lag_max={stat['behavior_lag_max']}",flush=True)
    elapsed=time.perf_counter()-started
    after_rl=model.evaluate(config.reward.effort_levels,config.train.max_generated_tokens)
    # Stage boundary: no retained RL trajectory is used for OPD. The simple
    # executable owns no remote workers; pending synthetic groups are discarded.
    discarded_for_stage=controller.pending_count
    if hasattr(controller,"cancel_pending"):
        controller.cancel_pending("end_of_rl_stage")
    del groups,task_meta,controller
    opd_losses=[]
    if config.train.opd_steps:
        from .distillation import full_vocabulary_reverse_kl
        opd_optimizer=torch.optim.Adam(model.parameters(),lr=config.train.learning_rate/5)
        for _ in range(config.train.opd_steps):
            contexts=[]
            answers=[]
            for symbol in range(8):
                effort=rng.choice(config.reward.effort_levels)
                prompt=model.prompt(symbol,effort)
                tokens,_=model.sample(prompt,max_tokens=config.train.max_generated_tokens,generator=generator)
                for offset in range(len(tokens)):
                    contexts.append(prompt+tokens[:offset])
                    answers.append(symbol)
            logits=model.logits(contexts)
            # Explicit synthetic oracle, never represented as a sourced teacher.
            teacher=torch.full_like(logits,-3.)
            teacher[torch.arange(len(contexts)),torch.tensor(answers)]=3.
            opd_optimizer.zero_grad(set_to_none=True)
            loss=full_vocabulary_reverse_kl(logits,teacher,torch.ones(len(contexts),dtype=torch.bool),
                student_tokenizer_fingerprint=model.tokenizer_fingerprint,
                teacher_tokenizer_fingerprint=model.tokenizer_fingerprint)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.train.gradient_clip)
            opd_optimizer.step()
            opd_losses.append(float(loss.detach().item()))
    final=model.evaluate(config.reward.effort_levels,config.train.max_generated_tokens)
    summary={"implementation":"FusionRL 0.1.0","environment":"synthetic-symbol-copy-v1",
        "evidence_scope":"CPU causal-policy integration test; no real LLM, MiMo environment, cluster throughput or held-out benchmark claim",
        "mode":config.schedule.mode,"async_execution":"deterministic delayed-queue simulation" if config.schedule.mode=="bounded_async" else "strict synchronous",
        "seed":config.train.seed,"rl_steps":config.train.steps,"sft_steps":config.train.sft_steps,
        "opd_steps":config.train.opd_steps,"initial":starting,"after_sft":after_sft,"after_rl":after_rl,"final":final,
        "observed_max_admitted_token_lag":max(m["behavior_lag_max"] for m in metrics),
        "discarded_pending_groups_at_stage_boundary":discarded_for_stage,
        "rl_elapsed_seconds":elapsed,"torch_version":torch.__version__,"numpy_version":np.__version__,
        "parameter_count":sum(p.numel() for p in model.parameters()),"opd_losses":opd_losses}
    (output/"config.json").write_text(json.dumps(config.to_dict(),indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (output/"metrics.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in metrics))
    (output/"rollout_example.json").write_text(json.dumps(trace,indent=2,allow_nan=False)+"\n")
    torch.save({"state_dict":model.state_dict(),"tokenizer_fingerprint":model.tokenizer_fingerprint,
                "config":config.to_dict(),"completed_rl_steps":config.train.steps},output/"checkpoint.pt")
    return summary

