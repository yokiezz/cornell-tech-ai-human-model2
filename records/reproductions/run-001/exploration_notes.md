# Training-only exploration review

Codex inspected the plot and all six seed-42 examples: human IDs 1384, 8677, 1295 and AI IDs 1342, 8766, 1273. Membership was checked against the canonical training IDs. Raw texts stay in ignored artifacts. User interpretation is pending.

Each class contains 4,000 training rows. Human and AI word-count means are 209.33 and 208.72; medians 204 and 209; interquartile ranges 167–251 and 151–263. Both span 75–350 words. There is substantial overlap, and the AI length distribution is broader. Length alone is not a reliable authorship rule.

Newline medians are 2 for human and 6 for AI; means 5.80 and 12.21, with maxima 35 and 170. The long AI tail affects the average, although the medians also differ. Paragraph formatting is one possible explanation; genre or source composition is another. The six examples include news/reference-style human passages and review/news-like/technical AI passages, with variation within both labels. These descriptions are impressions, not verified source metadata.

A proposed follow-up would compare newline distributions within annotated genres, or a separately approved training-only whitespace-normalization ablation. Neither was performed; no feature or label changes followed from these observations. This exploration occurred in this reproduction, not before the original source pilot.
