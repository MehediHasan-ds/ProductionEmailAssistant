# Model Comparison Summary

Data source: 12 evaluation scenarios across 5 categories (normal, complex, rare, failure, malicious), scored by 3 custom metric groups. Reports in reports/metrics_report_openrouter.csv and reports/metrics_report_gemini.csv.


## 1. Which model performed better?

GPT-OSS-120B via OpenRouter performed better overall, with a higher average score and better reliability.

| Metric | GPT-OSS-120B | Gemini-2.5-Flash-Lite |
|---|---|---|
| Scenarios completed | 12 of 12 | 11 of 12 |
| Overall average | 82.91 | 81.52 |
| Rule-based average | 0.92 | 0.88 |
| Judge average | 0.96 | 0.95 |
| Reference cosine | 0.93 | 0.92 |

OpenRouter won 8 of 12 head-to-head scenario comparisons. It scored higher on every normal case and on the complex payment reminder (84.03 vs 73.67, a 10 point gap). Gemini won 3 scenarios: the diplomatic decline (83.97 vs 79.76), the contradictory facts case (77.61 vs 72.72), and the prompt injection case (82.40 vs 81.79).

The key differentiator was tone matching. OpenRouter averaged 0.89 on rule-based tone match versus Gemini at 0.70. Gemini struggled to match diplomatic and empathetic tones, scoring 0.33 on tone match for both the onboarding welcome and the missing facts scenario. OpenRouter also maintained perfect professionalism (1.0 average from the judge) while Gemini dipped to 0.91, with the payment reminder scoring just 0.8.


## 2. Biggest failure mode of the lower-performing model

Gemini had two failure modes.

**Quality: placeholder leakage and tone mismatch.** On the complex payment reminder scenario (its lowest score at 73.67), Gemini produced an email containing a placeholder, scoring 0.0 on placeholder_leak. It also scored 0.8 on professionalism, fact integration, and clarity from the judge. The tone match score averaged just 0.70 across all scenarios, compared to OpenRouter at 0.89, indicating Gemini frequently failed to match the requested tone in its word choices.

**Infrastructure: rate limiting.** Gemini returned HTTP 429 on 2 of 12 scenarios, producing zero output for the exfiltration case and intermittently failing on others. The free tier of Google AI Studio limits requests per minute tightly enough that a 12-scenario evaluation (24 API calls) cannot complete without significant delays. This is an infrastructure constraint, not a quality issue, but it makes the model unreliable under sustained load.


## 3. Recommendation for production

Recommend GPT-OSS-120B via OpenRouter.

Justification from the custom metric data:

- **Overall quality:** 82.91 average versus 81.52. The gap is small but consistent across scenario categories.
- **Tone matching:** 0.89 versus 0.70. OpenRouter reliably produces text that matches the requested tone, which is the core value proposition of an email assistant.
- **Professionalism:** 1.0 (perfect) versus 0.91. OpenRouter never produced unprofessional output across 12 scenarios.
- **Placeholder safety:** 1.0 versus 0.91. OpenRouter never leaked a placeholder into the final email. Gemini leaked one in the payment scenario.
- **Safety on adversarial inputs:** OpenRouter completed both malicious scenarios (81.79 and 87.63) with no instruction override or secret leakage. Gemini completed one (82.40) but failed on the second due to rate limiting.
- **Reliability:** 12 of 12 scenarios completed (100 percent) versus 11 of 12 for Gemini. A model that rate-limits under evaluation load will rate-limit in production.

Gemini is a viable secondary or fallback provider. Its emails are semantically close to the reference (cosine 0.92) and its judge scores are competitive. But its tone matching weakness, placeholder leak risk exposure make it the weaker choice for a production email assistant.
