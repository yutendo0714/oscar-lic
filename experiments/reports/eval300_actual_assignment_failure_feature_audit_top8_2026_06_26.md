# eval300_actual_assignment_failure_feature_audit_top8_2026_06_26

Diagnostic-only enrichment of the actual assignment failure bank with deployable candidate features.
Held-out OCR outcomes define the strata but are not used as deployable inputs.

## Summary

- Cases: `75`
- Target counts: `{'keep_current_good_change': 4, 'keep_noop': 57, 'recover_shortlist_oracle_change': 13, 'reject_current_bad_change': 1}`
- Missing feature rows by role: `{}`

## Score Rank Diagnostic

| target | score | mean score | mean rank | rank<=1 | rank<=2 | rank<=4 |
|---|---|---:|---:|---:|---:|---:|
| recover_shortlist_oracle_change | `codeonly` | 0.07057154739168148 | 2.8333333333333335 | 0 | 2 | 6 |
| recover_shortlist_oracle_change | `diffcrop` | 0.032298804733584326 | 2.3333333333333335 | 0 | 4 | 6 |
| recover_shortlist_oracle_change | `latentwindow` | 0.04499269699166872 | 3.0 | 0 | 1 | 6 |
| recover_shortlist_oracle_change | `latentctx` | 0.03657776476868927 | 2.5 | 0 | 3 | 6 |
| recover_shortlist_oracle_change | `tabular_imgdiff` | 0.002361247094643397 | 5.384615384615385 | 0 | 0 | 5 |
| reject_current_bad_change | `codeonly` | 0.9999280373255411 | 1.0 | 1 | 1 | 1 |
| reject_current_bad_change | `diffcrop` | 0.9366579453150431 | 1.0 | 1 | 1 | 1 |
| reject_current_bad_change | `latentwindow` | 0.9993811845779419 | 1.0 | 1 | 1 | 1 |
| reject_current_bad_change | `latentctx` | 0.9767207900683085 | 1.0 | 1 | 1 | 1 |
| reject_current_bad_change | `tabular_imgdiff` | 0.8876417477925619 | 1.0 | 1 | 1 | 1 |
| keep_current_good_change | `codeonly` | 0.6837616507144163 | 1.5 | 3 | 3 | 4 |
| keep_current_good_change | `diffcrop` | 0.3202215927964289 | 2.0 | 1 | 3 | 4 |
| keep_current_good_change | `latentwindow` | 0.739475832508654 | 1.5 | 3 | 3 | 4 |
| keep_current_good_change | `latentctx` | 0.6515594911606968 | 1.5 | 3 | 3 | 4 |
| keep_current_good_change | `tabular_imgdiff` | 0.23607708956463577 | 2.25 | 1 | 3 | 4 |
| keep_noop | `codeonly` | 0.7817825978173556 | 1.368421052631579 | 44 | 52 | 57 |
| keep_noop | `diffcrop` | 0.8440252621494535 | 1.1228070175438596 | 50 | 57 | 57 |
| keep_noop | `latentwindow` | 0.792835981233345 | 1.3157894736842106 | 45 | 53 | 57 |
| keep_noop | `latentctx` | 0.707440198404539 | 1.4912280701754386 | 41 | 49 | 56 |
| keep_noop | `tabular_imgdiff` | 0.7537405907318291 | 1.087719298245614 | 52 | 57 | 57 |

## Feature Means

| target | count | feature | mean | min | max |
|---|---:|---|---:|---:|---:|
| recover_shortlist_oracle_change | 13 | `topk_rank` | 3.3846153846153846 | 1.0 | 7.0 |
| recover_shortlist_oracle_change | 13 | `assignment_relative_error` | 1.1164849538069506 | 1.0119526386260986 | 1.333662509918213 |
| recover_shortlist_oracle_change | 13 | `codebook_delta_l2` | 0.878066177551563 | 0.6486960053443909 | 1.3729530572891235 |
| recover_shortlist_oracle_change | 13 | `codebook_code_nearest_cosine` | 0.5465727780873959 | 0.11197589337825775 | 0.6853041052818298 |
| recover_shortlist_oracle_change | 13 | `img_variant_nearest_mse` | 1.0342066441528284e-06 | 4.977480898560316e-07 | 1.6144631445058621e-06 |
| recover_shortlist_oracle_change | 13 | `img_variant_nearest_changed_fraction` | 0.0006085060772560772 | 0.0 | 0.00244140625 |
| recover_shortlist_oracle_change | 13 | `img_variant_nearest_bbox_area_fraction` | 0.0738790994162088 | 0.0 | 0.40625 |
| recover_shortlist_oracle_change | 13 | `img_source_variant_mse` | 0.00025138261620528425 | 0.00014412276505026966 | 0.0003129505494143814 |
| recover_shortlist_oracle_change | 13 | `img_source_edge_density` | 0.17985237121936007 | 0.00725 | 0.2775 |
| reject_current_bad_change | 1 | `topk_rank` | 2.0 | 2.0 | 2.0 |
| reject_current_bad_change | 1 | `assignment_relative_error` | 1.0280495882034302 | 1.0280495882034302 | 1.0280495882034302 |
| reject_current_bad_change | 1 | `codebook_delta_l2` | 1.0231375694274902 | 1.0231375694274902 | 1.0231375694274902 |
| reject_current_bad_change | 1 | `codebook_code_nearest_cosine` | 0.39356574416160583 | 0.39356574416160583 | 0.39356574416160583 |
| reject_current_bad_change | 1 | `img_variant_nearest_mse` | 1.4517653426082688e-06 | 1.4517653426082688e-06 | 1.4517653426082688e-06 |
| reject_current_bad_change | 1 | `img_variant_nearest_changed_fraction` | 0.0026041666666666665 | 0.0026041666666666665 | 0.0026041666666666665 |
| reject_current_bad_change | 1 | `img_variant_nearest_bbox_area_fraction` | 0.10546875 | 0.10546875 | 0.10546875 |
| reject_current_bad_change | 1 | `img_source_variant_mse` | 0.00022806732158642262 | 0.00022806732158642262 | 0.00022806732158642262 |
| reject_current_bad_change | 1 | `img_source_edge_density` | 0.1998663101604278 | 0.1998663101604278 | 0.1998663101604278 |
| keep_current_good_change | 4 | `topk_rank` | 1.75 | 1.0 | 3.0 |
| keep_current_good_change | 4 | `assignment_relative_error` | 1.0550171732902527 | 1.0057493448257446 | 1.1039564609527588 |
| keep_current_good_change | 4 | `codebook_delta_l2` | 0.9333712011575699 | 0.6486960053443909 | 1.1258330345153809 |
| keep_current_good_change | 4 | `codebook_code_nearest_cosine` | 0.6219000071287155 | 0.5739590525627136 | 0.6853041052818298 |
| keep_current_good_change | 4 | `img_variant_nearest_mse` | 1.7628311752559966e-06 | 5.206330797591363e-07 | 2.42695114138769e-06 |
| keep_current_good_change | 4 | `img_variant_nearest_changed_fraction` | 0.0015340169270833333 | 0.0003255208333333333 | 0.003125 |
| keep_current_good_change | 4 | `img_variant_nearest_bbox_area_fraction` | 0.14651692708333333 | 0.0003255208333333333 | 0.51025390625 |
| keep_current_good_change | 4 | `img_source_variant_mse` | 0.00027980674713035114 | 0.00022839770826976746 | 0.00031068286625668406 |
| keep_current_good_change | 4 | `img_source_edge_density` | 0.15221105304358948 | 0.06287350597609562 | 0.23553054662379422 |
| keep_noop | 57 | `topk_rank` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `assignment_relative_error` | 1.0 | 1.0 | 1.0 |
| keep_noop | 57 | `codebook_delta_l2` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `codebook_code_nearest_cosine` | 1.0000000428735165 | 0.9999998807907104 | 1.0000001192092896 |
| keep_noop | 57 | `img_variant_nearest_mse` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_changed_fraction` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_bbox_area_fraction` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_source_variant_mse` | 0.000239008255384601 | 7.948950951686129e-05 | 0.000434441608376801 |
| keep_noop | 57 | `img_source_edge_density` | 0.1417523429405146 | 0.0006684491978609625 | 0.2805466237942122 |

## Priority Rows

| target | key | source | ref | role | codes n/c/s/o | OCR deltas c/s/s-c | current gate | codeonly score/rank | diffcrop score/rank | latentwindow score/rank | latentctx score/rank | tabular_imgdiff score/rank |
|---|---|---|---|---|---|---:|---:|---|---|---|---|---|
| recover_shortlist_oracle_change | 1/9/0 | iam_words | `own` | shortlist | 62/62/31/31 | 0/-1/-1 | 0.9999996423721313 | 0.0007200828489051977/3 | 1.1282285053463662e-05/3 | 0.00216813357042156/3 | 0.009455893154760512/3 | 2.2854611307538485e-10/4 |
| recover_shortlist_oracle_change | 1/12/1 | iam_words | `Lawrence` | shortlist | 2/2/49/49 | 0/-3/-3 | 0.9999980926513672 | None/null | None/null | None/null | None/null | 2.087778613172914e-07/5 |
| recover_shortlist_oracle_change | 1/32/0 | iam_words | `predetermined` | shortlist | 62/62/50/50 | 0/-1/-1 | 0.9999998807907104 | 0.2764359113449852/3 | 0.00021416941126517486/2 | 0.21631227975012735/3 | 0.02440854177499811/3 | 2.846195935185536e-12/8 |
| recover_shortlist_oracle_change | 1/51/1 | iam_words | `THE` | shortlist | 59/59/56/56 | 0/-1/-1 | 0.9994057416915894 | None/null | None/null | None/null | None/null | 9.759263360923944e-12/4 |
| recover_shortlist_oracle_change | 1/53/1 | icdar2015 | `Accessories` | shortlist | 32/32/43/43 | 0/-1/-1 | 0.9606596827507019 | None/null | None/null | None/null | None/null | 1.2307241406873791e-15/8 |
| recover_shortlist_oracle_change | 2/4/0 | iam_words | `own` | shortlist | 18/18/36/36 | 0/-1/-1 | 0.555584728717804 | None/null | None/null | None/null | None/null | 4.938728607207603e-12/8 |
| recover_shortlist_oracle_change | 2/5/4 | iam_words | `Lawrence` | shortlist | 34/34/2/2 | 0/-1/-1 | 0.0004797042638529092 | None/null | None/null | None/null | None/null | 1.1517624297973129e-10/6 |
| recover_shortlist_oracle_change | 2/34/0 | icdar2015 | `finest` | shortlist | 18/18/1/1 | 0/-1/-1 | 0.9803978800773621 | None/null | None/null | None/null | None/null | 0.0002312917640665546/4 |
| recover_shortlist_oracle_change | 2/39/1 | iam_words | `predetermined` | shortlist | 18/18/1/1 | 0/-1/-1 | 0.04280534014105797 | 0.1462503540678881/2 | 0.0006027169537598335/2 | 0.05096551218593959/3 | 0.15685543976724148/2 | 3.2103518350368176e-06/5 |
| recover_shortlist_oracle_change | 2/43/4 | iam_words | `understand` | shortlist | 7/7/21/21 | 0/-1/-1 | 1.0 | 2.1615401540960495e-05/2 | 1.5653586896392905e-09/3 | 8.508050332996694e-06/3 | 0.02868724218569696/2 | 5.350013768058771e-15/3 |
| recover_shortlist_oracle_change | 2/50/0 | iam_words | `text` | shortlist | 18/18/46/46 | 0/-1/-1 | 0.08042218536138535 | 1.094514326875166e-06/4 | 0.006629755861164692/2 | 9.299531397270282e-05/4 | 3.5405392433555484e-05/3 | 0.030460462808453787/3 |
| recover_shortlist_oracle_change | 2/53/1 | synthtext_words | `vista"` | shortlist | 45/45/20/20 | 0/-1/-1 | 0.9999997615814209 | 2.2617244255229707e-07/3 | 0.18633490232490413/2 | 0.0004087530792181345/2 | 2.4066337005024252e-05/2 | 6.3245944649826796e-12/7 |
| recover_shortlist_oracle_change | 2/69/1 | icdar2013 | `COSTA` | shortlist | 1/1/25/25 | 0/-1/-1 | 0.004869639873504639 | None/null | None/null | None/null | None/null | 1.0381605497489232e-06/5 |
| reject_current_bad_change | 2/51/1 | iam_words | `from` | current_bad | 2/48/2/2 | 1/0/-1 | 1.0 | 0.9999280373255411/1 | 0.9366579453150431/1 | 0.9993811845779419/1 | 0.9767207900683085/1 | 0.8876417477925619/1 |
| keep_current_good_change | 1/63/2 | icdar2015 | `heart` | current | 53/19/19/19 | -3/-3/0 | 0.9999980926513672 | 0.9805116852124532/1 | 2.219838734163204e-06/2 | 0.9776678880055746/1 | 0.9697806437810262/1 | 0.00011581578386691642/2 |
| keep_current_good_change | 2/6/5 | icdar2013 | `SLUSH` | current | 34/62/62/62 | -2/-2/0 | 1.0 | 0.7669759293397268/1 | 0.9147194027900696/1 | 0.9869868954022726/1 | 0.6637515034526587/1 | 0.9208289583524069/1 |
| keep_current_good_change | 2/13/1 | iam_words | `Mauro's` | current | 46/18/18/18 | -3/-3/0 | 0.9526782035827637 | 0.9875056743621826/1 | 0.03268661066734543/2 | 0.9815435806910197/1 | 0.9726060628890991/1 | 0.00012853452877418606/2 |
| keep_current_good_change | 2/64/0 | icdar2013 | `JOHN` | current | 18/54/54/54 | -1/-1/0 | 0.9999173879623413 | 5.331394330217639e-05/3 | 0.33347813788956654/3 | 0.011704965935749101/3 | 9.97545200031406e-05/3 | 0.023235049593495205/4 |

## Contact Sheets

- recover_shortlist_oracle_change: `experiments/figures/eval300_actual_assignment_failure_feature_audit_top8_2026_06_26/recover_shortlist_oracle_change.png`
- reject_current_bad_change: `experiments/figures/eval300_actual_assignment_failure_feature_audit_top8_2026_06_26/reject_current_bad_change.png`
- keep_current_good_change: `experiments/figures/eval300_actual_assignment_failure_feature_audit_top8_2026_06_26/keep_current_good_change.png`

## Interpretation

- This audit should guide the next verifier design, not directly tune thresholds on the 75 held-out groups.
- The recover target is a high-precision selection problem inside the verified shortlist; the reject target is the abstention floor.
- If current deployable scores rank recover targets highly but also rank the reject case highly, add evidence rather than sweep thresholds.
