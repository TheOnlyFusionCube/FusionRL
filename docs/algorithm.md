# The FusionRL learning rule

The source locators below refer to [SOURCES.md](../SOURCES.md). Each optimizer update uses B homogeneous groups of N original executions. Let i index an execution. Let T_i be its admitted, provenance-checked, deduplicated target tokens. Each target u retains token y_u, original context c_u, behavior probability p_roll,u and actual generating policy version v_u.

## 1. Validate before assigning credit

The task contract defines an environment, an assessment rule and an immutable configuration key. Complete all original replicate slots. Keep a valid zero score as zero. If a score is unavailable because the environment or evaluator failed, quarantine the whole group. Confirmed verifier exploitation changes the effective binary outcome to zero before group statistics, GRS products or GAR ranking.

Externally produced rubric grades and rankings are bound to ordered execution IDs and assessed artifact IDs. `pipeline.prepare_step` checks both and the conditioning key. It supports different grading routes across groups in a single update while keeping their relative comparisons separate.

## 2. Select one grading route per group

### Binary or calibrated scalar

For binary verification use R_i in {0,1}. For an explicitly calibrated continuous score use R_i in [0,1] and declare `scalar`. In both cases:

$$A_i=R_i-\frac{1}{N}\sum_{j=1}^{N}R_j.$$

There is no division by group standard deviation. Continuous grades are not silently converted to binary passes. Calibration is fixed by the task/rubric contract, not refit independently in each group.

### GRS

With binary test outcome R_test and bounded rubric scores S_sol and S_beh:

$$R_i=R_i^{\mathrm{test}}S_i^{\mathrm{sol}}S_i^{\mathrm{beh}},\qquad A_i=R_i-\operatorname{mean}_j R_j.$$

This follows the reward construction in MiMo §4.3.1, Eq. 2. Different grades can provide a learning signal even if every execution passes its tests. Unusable required GRS grades invalidate that supervision record; no placeholder grade is substituted.

### GAR

Start from binary corrected outcomes and A_i=R_i-mean(R). Let P be the passing set. For mixed-outcome groups with usable f_i in (0,1]:

$$\lambda=\frac{\sum_{j\in P}A_j}{\sum_{j\in P}f_jA_j},\quad
\lambda_c=\min(\lambda,\lambda_{\max}),$$

$$B_i=\begin{cases}\lambda_cf_iA_i&i\in P,\\A_i&i\notin P,\end{cases}
\qquad A_i^{\mathrm{GAR}}=B_i-\operatorname{mean}_jB_j.$$

MiMo §4.3.2, Eq. 3 supplies this redistribution and practical recentering. With no cap, the positive advantage mass is preserved and failures are unchanged. With a binding cap, the subsequent recentering changes both passes and failures; claiming unchanged failures would be incorrect. The implementation's default cap 2.0 is a visible engineering choice, not a sourced universal optimum. Missing or unusable rankings fall back to the corrected binary advantages. Unanimous binary groups have no mass to redistribute.

GAR rejects nonzero additive cost penalties. It also rejects GRS channels. Concision can be one declared feature of its passing-solution quality judgment.

## 3. Optional cost channel for binary, GRS or scalar routes

Choose exactly one controller. A cost vector C_i is centered within the same group and added to its advantage. For these routes, this is equivalent to centering R_i+C_i. It must not be normalized against a different effort group.

DeepSeek §5.1.4, Eqs. 9–10 gives the effort-conditioned deduction:

$$C_i=-\min\left(C_{\max},\ k_0e^{-(b-b_{\min})/\tau}\frac{L_i^{\mathrm{reason}}}{L_{\mathrm{norm}}}\right).$$

All executions share the actual effort-conditioning value b. Count generated reasoning tokens, excluding prompt/tool/environment tokens. In the synthetic demo the designated special output symbols serve only as a transparent stand-in for this measurement.

The alternative MiMo §4.3.3, Eq. 4 compares successful output lengths with a successful-length quantile. `mimo_length_penalties` implements its strict pass-rate gate and bounded ramp. It needs explicit binary success flags even when the primary reward is scalar. Compute those flags after confirmed-hack correction. Supplying an arbitrary continuous score as the success indicator is rejected.

Do not drop a group merely because its raw tests all agree. GRS and cost shaping can make the final advantages nonzero. The reference trainer records zero-signal groups and preserves the dispatch-defined batch rather than refill from conveniently faster groups.

## 4. Freeze the admitted target set

Exclude non-policy provenance, then excessive actual-token staleness, then apply the path cap and generation-identity deduplication. Preserve original contexts for all retained targets. Let L_i=|T_i|. The shared target set controls the sequence statistic and the denominator; a token cannot enter one but not the other.

FusionRL's deterministic path selection and stale-token masking define a modified admitted-data objective. They do not establish unbiased recovery of the full unfiltered execution tree. All original execution slots and their assessed rewards remain, including slots with L_i=0.

## 5. One execution policy loss

FusionRL uses QwenGyre Appendix B, Eqs. 26–30 as its common optimizer. Define:

$$\rho_u(\theta)=\frac{\pi_\theta(y_u\mid c_u)}{p_{\mathrm{roll},u}},\qquad
s_i=\operatorname{sg}\left[\exp\left(\frac{1}{L_i}\sum_{u\in T_i}\log\rho_u\right)\right].$$

The shared clipping decision is:

$$m_i=\begin{cases}
0&A_i>0\ \land\ s_i>1.005,\\
0&A_i<0\ \land\ s_i<0.995,\\
1&\text{otherwise}.
\end{cases}$$

Use detached token importance weights and a one-sided loss cap:

$$w_u=\operatorname{sg}(\min(\rho_u,5)),\qquad
\ell_u=\min\{100,-m_iw_u\operatorname{sg}(A_i)\log\pi_\theta(y_u\mid c_u)\}.$$

Then:

$$\mathcal L_i=\begin{cases}\frac{1}{L_i}\sum_{u\in T_i}\ell_u&L_i>0,\\0&L_i=0,\end{cases}
\qquad \mathcal L=\frac{1}{BN}\sum_i\mathcal L_i.$$

The sequence statistic controls only the clipping decision. It is not another multiplicative importance ratio. The capped token weight is detached, so gradients flow through the final current-policy log probability. The loss is capped above, not by absolute magnitude. These distinctions are covered by independent value and gradient tests.

The implementation compares the mean log ratio with log clipping thresholds to avoid unnecessary exponentiation. It caps the exponent before computing token weights and carries reductions in at least FP32. The learner does not round stored behavior probabilities down to the model's BF16/FP16 output dtype.

## 6. Distributed reduction and update

Every microbatch must use the original global execution count. Entire execution rows stay together; splitting one execution requires a prior global reduction for its target count and clipping statistic. If the training framework averages gradients across D ranks, multiply the local execution sum divided by B*N by D exactly once. Do not additionally average by the number of microbatches or by surviving token counts.

The pinned FSDP integration's actual accumulation loop was inspected and tested against this convention. Other engines need their own reduction adapter. Token-level auxiliary advantage edits and additional rollout-IS/PPO/GSPO/KL/entropy losses are disabled in the supplied integration.

The CPU trainer skips the optimizer when there is no eligible signal or the complete gradient is zero, preventing momentum-only changes on wholly clipped data. Its controller still records a consumed batch and a publication version; a real checkpoint ledger should separately record weight digest and update count.

## 7. Optional OPD

At a clean stage boundary, cancel pending RL work and sample fresh student prefixes. With aligned full-vocabulary logits, `full_vocabulary_reverse_kl` computes mean KL(student || teacher), detaching the teacher. This divergence is FusionRL's explicit choice. The tokenizer fingerprint must cover the token-ID mapping, normalization, special tokens and chat-template contract; equal vocabulary sizes are insufficient. A model architecture match is unnecessary when the tokenization and conditioning contract does match.

