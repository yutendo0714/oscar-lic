# eval300_actual_assignment_imgstroke_audit_2026_06_26

Diagnostic-only audit of existing imgstroke/projection features on the N076 actual assignment failure bank.
Held-out OCR categories define strata only; no selector is trained or promoted.

## Summary

- Cases: `75`
- Target counts: `{'improve_first_stage_shortlist_recall': 2, 'keep_current_good_change': 4, 'keep_noop': 57, 'recover_shortlist_oracle_change': 11, 'reject_current_bad_change': 1}`
- Feature count: `130`
- Standalone reject-separation features: `16`

## Selected Feature Means

| target | count | feature | mean | min | max |
|---|---:|---|---:|---:|---:|
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_changed_fraction` | 0.0006747543413026367 | 0.0 | 0.00244140625 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_mse` | 1.0616853335870309e-06 | 4.977480898560316e-07 | 1.6144631445058621e-06 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_dark_changed_fraction` | 0.0026351082646711307 | 0.0 | 0.012903225608170033 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_edge_changed_fraction` | 0.001607297485778955 | 0.0 | 0.007102272938936949 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_row_projection_abs_mean` | 8.194959214465185e-05 | 5.033425986766815e-05 | 0.0001665893942117691 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_col_projection_abs_mean` | 0.00013404782393164086 | 7.659303082618862e-05 | 0.00020220577425789088 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_row_dark_projection_abs_mean` | 0.00013236046916889873 | 0.0 | 0.0007102265954017639 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_col_dark_projection_abs_mean` | 0.0001323605406055735 | 0.0 | 0.0007102272938936949 |
| recover_shortlist_oracle_change | 11 | `img_variant_nearest_bbox_area_fraction` | 0.08578023538961038 | 0.0 | 0.40625 |
| recover_shortlist_oracle_change | 11 | `img_source_variant_dark_union_abs_mean` | 0.01732027894732627 | 0.008765517733991146 | 0.02158706821501255 |
| recover_shortlist_oracle_change | 11 | `img_source_variant_edge_union_abs_mean` | 0.018408219414678486 | 0.015226338058710098 | 0.02023262344300747 |
| recover_shortlist_oracle_change | 11 | `img_source_variant_row_dark_projection_abs_mean` | 0.006596427664838054 | 0.002790180966258049 | 0.01806640625 |
| recover_shortlist_oracle_change | 11 | `img_source_variant_col_dark_projection_abs_mean` | 0.007539938754317435 | 0.00341796875 | 0.01513671875 |
| recover_shortlist_oracle_change | 11 | `img_source_edge_density` | 0.18572306208986278 | 0.00725 | 0.2775 |
| recover_shortlist_oracle_change | 11 | `img_source_dark050_fraction` | 0.353505325167257 | 0.09609375 | 0.98388671875 |
| reject_current_bad_change | 1 | `img_variant_nearest_changed_fraction` | 0.0026041666666666665 | 0.0026041666666666665 | 0.0026041666666666665 |
| reject_current_bad_change | 1 | `img_variant_nearest_mse` | 1.4517653426082688e-06 | 1.4517653426082688e-06 | 1.4517653426082688e-06 |
| reject_current_bad_change | 1 | `img_variant_nearest_dark_changed_fraction` | 0.022727273404598236 | 0.022727273404598236 | 0.022727273404598236 |
| reject_current_bad_change | 1 | `img_variant_nearest_edge_changed_fraction` | 0.006289307959377766 | 0.006289307959377766 | 0.006289307959377766 |
| reject_current_bad_change | 1 | `img_variant_nearest_row_projection_abs_mean` | 0.00011488795280456543 | 0.00011488795280456543 | 0.00011488795280456543 |
| reject_current_bad_change | 1 | `img_variant_nearest_col_projection_abs_mean` | 0.00016595423221588135 | 0.00016595423221588135 | 0.00016595423221588135 |
| reject_current_bad_change | 1 | `img_variant_nearest_row_dark_projection_abs_mean` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | 1 | `img_variant_nearest_col_dark_projection_abs_mean` | 0.0 | 0.0 | 0.0 |
| reject_current_bad_change | 1 | `img_variant_nearest_bbox_area_fraction` | 0.10546875 | 0.10546875 | 0.10546875 |
| reject_current_bad_change | 1 | `img_source_variant_dark_union_abs_mean` | 0.019463669508695602 | 0.019463669508695602 | 0.019463669508695602 |
| reject_current_bad_change | 1 | `img_source_variant_edge_union_abs_mean` | 0.018446797505021095 | 0.018446797505021095 | 0.018446797505021095 |
| reject_current_bad_change | 1 | `img_source_variant_row_dark_projection_abs_mean` | 0.0045572929084300995 | 0.0045572929084300995 | 0.0045572929084300995 |
| reject_current_bad_change | 1 | `img_source_variant_col_dark_projection_abs_mean` | 0.0045572915114462376 | 0.0045572915114462376 | 0.0045572915114462376 |
| reject_current_bad_change | 1 | `img_source_edge_density` | 0.1998663101604278 | 0.1998663101604278 | 0.1998663101604278 |
| reject_current_bad_change | 1 | `img_source_dark050_fraction` | 0.08658854166666667 | 0.08658854166666667 | 0.08658854166666667 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_changed_fraction` | 0.000244140625 | 0.0 | 0.00048828125 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_mse` | 8.830738522647152e-07 | 8.650519021102809e-07 | 9.010958024191495e-07 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_dark_changed_fraction` | 0.0005414184997789562 | 0.0 | 0.0010828369995579123 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_edge_changed_fraction` | 0.000445632787887007 | 0.0 | 0.000891265575774014 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_row_projection_abs_mean` | 6.523076444864273e-05 | 6.510317325592041e-05 | 6.535835564136505e-05 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_col_projection_abs_mean` | 9.880730431177653e-05 | 9.80431868811138e-05 | 9.957142174243927e-05 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_row_dark_projection_abs_mean` | 0.0001220703125 | 0.0 | 0.000244140625 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_col_dark_projection_abs_mean` | 0.0001220703125 | 0.0 | 0.000244140625 |
| improve_first_stage_shortlist_recall | 2 | `img_variant_nearest_bbox_area_fraction` | 0.0084228515625 | 0.0 | 0.016845703125 |
| improve_first_stage_shortlist_recall | 2 | `img_source_variant_dark_union_abs_mean` | 0.016440399922430515 | 0.01331401988863945 | 0.01956677995622158 |
| improve_first_stage_shortlist_recall | 2 | `img_source_variant_edge_union_abs_mean` | 0.018461667001247406 | 0.017760492861270905 | 0.019162841141223907 |
| improve_first_stage_shortlist_recall | 2 | `img_source_variant_row_dark_projection_abs_mean` | 0.00759277306497097 | 0.00468749925494194 | 0.010498046875 |
| improve_first_stage_shortlist_recall | 2 | `img_source_variant_col_dark_projection_abs_mean` | 0.011059570359066129 | 0.0062500000931322575 | 0.015869140625 |
| improve_first_stage_shortlist_recall | 2 | `img_source_edge_density` | 0.14756357143159562 | 0.14143426294820718 | 0.15369287991498407 |
| improve_first_stage_shortlist_recall | 2 | `img_source_dark050_fraction` | 0.2693603515625 | 0.09609375 | 0.442626953125 |
| keep_current_good_change | 4 | `img_variant_nearest_changed_fraction` | 0.0015340169270833333 | 0.0003255208333333333 | 0.003125 |
| keep_current_good_change | 4 | `img_variant_nearest_mse` | 1.7628311752559966e-06 | 5.206330797591363e-07 | 2.42695114138769e-06 |
| keep_current_good_change | 4 | `img_variant_nearest_dark_changed_fraction` | 0.006120304838987067 | 0.0 | 0.019607843831181526 |
| keep_current_good_change | 4 | `img_variant_nearest_edge_changed_fraction` | 0.0028305436426308006 | 0.0 | 0.0074487896636128426 |
| keep_current_good_change | 4 | `img_variant_nearest_row_projection_abs_mean` | 0.00012830598279833794 | 5.6162476539611816e-05 | 0.0001723356544971466 |
| keep_current_good_change | 4 | `img_variant_nearest_col_projection_abs_mean` | 0.00022140593500807881 | 8.425302803516388e-05 | 0.0003255251795053482 |
| keep_current_good_change | 4 | `img_variant_nearest_row_dark_projection_abs_mean` | 0.0006835937383584678 | 0.0 | 0.001953125 |
| keep_current_good_change | 4 | `img_variant_nearest_col_dark_projection_abs_mean` | 0.000683593752910383 | 0.0 | 0.001953125 |
| keep_current_good_change | 4 | `img_variant_nearest_bbox_area_fraction` | 0.14651692708333333 | 0.0003255208333333333 | 0.51025390625 |
| keep_current_good_change | 4 | `img_source_variant_dark_union_abs_mean` | 0.01954312901943922 | 0.012560892850160599 | 0.026143819093704224 |
| keep_current_good_change | 4 | `img_source_variant_edge_union_abs_mean` | 0.016765001229941845 | 0.014341048896312714 | 0.020348450168967247 |
| keep_current_good_change | 4 | `img_source_variant_row_dark_projection_abs_mean` | 0.012320963898673654 | 0.00146484375 | 0.02978515625 |
| keep_current_good_change | 4 | `img_source_variant_col_dark_projection_abs_mean` | 0.01651204435620457 | 0.00146484375 | 0.0478515625 |
| keep_current_good_change | 4 | `img_source_edge_density` | 0.15221105304358948 | 0.06287350597609562 | 0.23553054662379422 |
| keep_current_good_change | 4 | `img_source_dark050_fraction` | 0.24093424479166667 | 0.00146484375 | 0.70263671875 |
| keep_noop | 57 | `img_variant_nearest_changed_fraction` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_mse` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_dark_changed_fraction` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_edge_changed_fraction` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_row_projection_abs_mean` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_col_projection_abs_mean` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_row_dark_projection_abs_mean` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_col_dark_projection_abs_mean` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_variant_nearest_bbox_area_fraction` | 0.0 | 0.0 | 0.0 |
| keep_noop | 57 | `img_source_variant_dark_union_abs_mean` | 0.01594772173516583 | 0.0 | 0.0300653874874115 |
| keep_noop | 57 | `img_source_variant_edge_union_abs_mean` | 0.01777538550984964 | 0.010499279946088791 | 0.02956259250640869 |
| keep_noop | 57 | `img_source_variant_row_dark_projection_abs_mean` | 0.006315202738174744 | 0.0 | 0.0185546875 |
| keep_noop | 57 | `img_source_variant_col_dark_projection_abs_mean` | 0.007178325271397306 | 0.0 | 0.033447265625 |
| keep_noop | 57 | `img_source_edge_density` | 0.1417523429405146 | 0.0006684491978609625 | 0.2805466237942122 |
| keep_noop | 57 | `img_source_dark050_fraction` | 0.28157566727764344 | 0.0 | 1.0 |

## Reject Separation Probe

| feature | reject | recover min/mean/max | good min/max | first-stage mean | direction | good collisions | abs z |
|---|---:|---|---|---:|---|---:|---:|
| `img_variant_nearest_dark_changed_fraction` | 0.022727273404598236 | 0.0/0.0026351082646711307/0.012903225608170033 | 0.0/0.019607843831181526 | 0.0005414184997789562 | above | 0 | 5.5339951600662225 |
| `img_source_variant_row_projection_abs_mean` | 0.003530927002429962 | 0.001870783045887947/0.0026074176738885317/0.0032342523336410522 | 0.0027860822156071663/0.003336407244205475 | 0.0025236732326447964 | above | 0 | 2.2900619718173423 |
| `img_source_nearest_row_projection_abs_mean` | 0.003507956862449646 | 0.001918654888868332/0.0026156891518357124/0.003262847661972046 | 0.0026740627363324165/0.0032996535301208496 | 0.002537206280976534 | above | 0 | 2.199533868710674 |
| `img_variant_nearest_edge_density_delta` | -0.0006684491978609652 | -0.0003980891719745361/0.0002194234494798058/0.0010000000000000009 | -0.000622509960159362/0.0005000000000000004 | -6.641870350690415e-05 | below | 0 | 2.075884557227345 |
| `img_source_variant_signed_abs_ratio` | 0.34310200810432434 | 0.004154857713729143/0.16688109820031308/0.31296879053115845 | 0.1477755606174469/0.24942034482955933 | 0.18942155875265598 | above | 0 | 1.6902458674096508 |
| `img_source_nearest_signed_abs_ratio` | 0.3431263566017151 | 0.0037093968130648136/0.16826538745821876/0.31509825587272644 | 0.13633467257022858/0.2440664917230606 | 0.19033157266676426 | above | 0 | 1.6852705308907172 |
| `img_source_nearest_changed_fraction` | 0.4342447916666667 | 0.45989583333333334/0.5560018432088745/0.6764914772727273 | 0.4482421875/0.710205078125 | 0.554296875 | below | 0 | 1.5195703194468566 |
| `img_source_variant_changed_fraction` | 0.43359375 | 0.46041666666666664/0.5558106045233175/0.6775568181818182 | 0.4495442708333333/0.710693359375 | 0.5539632161458333 | below | 0 | 1.51482526933521 |
| `img_source_variant_signed_mean` | 0.003168402472510934 | 3.6382058169692755e-05/0.0017812415753724054/0.0029401551000773907 | 0.0019971663132309914/0.002883731387555599 | 0.0018563624180387706 | above | 0 | 1.3958622152944882 |
| `img_source_nearest_signed_mean` | 0.0031607430428266525 | 3.2552401535212994e-05/0.001794930379467339/0.00296466494910419 | 0.0018411080818623304/0.002787990029901266 | 0.0018691917066462338 | above | 0 | 1.383118978961703 |
| `img_nearest_ink_center_y_fraction` | 0.4105571847507331 | 0.4351716961498439/0.5457689472466628/0.7691559876270437 | 0.46790162887256465/0.6704859274928021 | 0.6771191702109284 | below | 0 | 1.3804379752083376 |
| `img_variant_ink_center_y_fraction` | 0.4105571847507331 | 0.4351716961498439/0.5456365832777382/0.7691559876270437 | 0.4693232131562302/0.6704859274928021 | 0.6771989836978977 | below | 0 | 1.3801044275781784 |
| `img_source_ink_center_y_fraction` | 0.4162017948096046 | 0.4273159636062862/0.5440047682553275/0.7664131480024479 | 0.45099255583126546/0.6989247311827957 | 0.6750682183974306 | below | 0 | 1.2973009286380084 |
| `img_source_mean` | 0.8947814106941223 | 0.31135302782058716/0.707791266116229/0.8919536471366882 | 0.4130916893482208/0.8746568560600281 | 0.6691405177116394 | above | 0 | 0.9051320601737254 |
| `img_nearest_mean` | 0.8916208148002625 | 0.31118834018707275/0.7059963535178791/0.8889890313148499 | 0.41125059127807617/0.8725091814994812 | 0.6672713160514832 | above | 0 | 0.9015772927691871 |
| `img_variant_mean` | 0.8916130661964417 | 0.3112592101097107/0.7060100571675734/0.8890135288238525 | 0.4110945463180542/0.8724601864814758 | 0.6672841608524323 | above | 0 | 0.9015411331847429 |
| `img_variant_nearest_bbox_width_fraction` | 0.5625 | 0.0/0.12164748130657221/0.43333333333333335 | 0.010416666666666666/0.7421875 | 0.08984375 | above | 1 | 2.874137190373931 |
| `img_variant_nearest_changed_fraction` | 0.0026041666666666665 | 0.0/0.0006747543413026367/0.00244140625 | 0.0003255208333333333/0.003125 | 0.000244140625 | above | 1 | 2.4837255342990274 |
| `img_variant_nearest_a_dark_abs_mean` | 0.0008318479522131383 | 0.00016177444194909185/0.00041089534748938274/0.0008222643518820405 | 0.0/0.0011149559868499637 | 0.00030061148572713137 | above | 1 | 1.9610971432896915 |
| `img_variant_nearest_b_dark_abs_mean` | 0.0008318479522131383 | 0.00016177444194909185/0.0004114130781751803/0.0008222643518820405 | 0.0/0.0010871675331145525 | 0.00030161214817781 | above | 1 | 1.9599518303699233 |
| `img_variant_nearest_dark_union_abs_mean` | 0.0008318479522131383 | 0.00016177444194909185/0.00041233381753872066/0.0008222643518820405 | 0.0/0.0011149559868499637 | 0.00030161214817781 | above | 1 | 1.9557660534996901 |
| `img_source_variant_dark_changed_fraction` | 0.8676470518112183 | 0.5488165616989136/0.7549828399311412/0.8495298027992249 | 0.6870681047439575/1.0 | 0.7270143628120422 | above | 1 | 1.0733888110336247 |
| `img_source_nearest_dark_changed_fraction` | 0.8529411554336548 | 0.5522682666778564/0.7560213695872914/0.8401253819465637 | 0.6876027584075928/1.0 | 0.7309410572052002 | above | 1 | 0.9318731132086066 |
| `img_nearest_dark050_fraction` | 0.0859375 | 0.09505208333333333/0.354577276416765/0.982421875 | 0.0/0.71923828125 | 0.2729899088541667 | below | 2 | 0.8140442770864998 |

## Interpretation

- This closes only the standalone imgstroke/projection-veto question.
- Features that separate the single reject from recover rows must still be treated as weak evidence because there is only one reject sample and good changes may collide.
- Use these features only as auxiliary inputs inside a stricter verifier with external calibration, not as a direct threshold rule.
