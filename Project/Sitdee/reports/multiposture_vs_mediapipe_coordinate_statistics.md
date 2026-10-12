# MultiPosture vs MediaPipe coordinate statistics

## Scope

Statistics cover all `4794` local MultiPosture rows and the first `300` frames of each DriveNet clip (`900` external rows total). DriveNet frames were resized to `640x480` to mirror the paper's reported source preprocessing. This is a distribution study; no external label was used to select an adapter.

| Representation | Landmark | Axis | Min | Max | Mean | Std | Median | P10 | P25 | P75 | P90 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MultiPosture local CSV` | `nose` | x | -0.39174 | 0.44991 | 0.00508 | 0.11950 | 0.00336 | -0.06339 | -0.01686 | 0.02953 | 0.08116 |
| `MultiPosture local CSV` | `nose` | y | -0.62062 | 0.22378 | -0.45220 | 0.11909 | -0.48697 | -0.54707 | -0.51939 | -0.44186 | -0.31542 |
| `MultiPosture local CSV` | `nose` | z | -0.81091 | 0.08413 | -0.34346 | 0.15027 | -0.34056 | -0.55254 | -0.42766 | -0.24755 | -0.16041 |
| `MultiPosture local CSV` | `left_eye_inner` | x | -0.38929 | 0.46988 | 0.01921 | 0.12085 | 0.01807 | -0.04961 | -0.00238 | 0.04322 | 0.09186 |
| `MultiPosture local CSV` | `left_eye_inner` | y | -0.64065 | 0.19406 | -0.47824 | 0.11810 | -0.51349 | -0.57082 | -0.54472 | -0.47026 | -0.33851 |
| `MultiPosture local CSV` | `left_eye_inner` | z | -0.80121 | 0.10101 | -0.33669 | 0.15700 | -0.33370 | -0.55937 | -0.42261 | -0.23833 | -0.14868 |
| `MultiPosture local CSV` | `left_eye` | x | -0.38875 | 0.47136 | 0.02199 | 0.12096 | 0.02112 | -0.04650 | 0.00080 | 0.04660 | 0.09509 |
| `MultiPosture local CSV` | `left_eye` | y | -0.64163 | 0.19515 | -0.47707 | 0.11894 | -0.51251 | -0.57024 | -0.54345 | -0.46984 | -0.33773 |
| `MultiPosture local CSV` | `left_eye` | z | -0.78163 | 0.10831 | -0.32642 | 0.15597 | -0.32319 | -0.54783 | -0.41187 | -0.22841 | -0.13902 |
| `MultiPosture local CSV` | `left_eye_outer` | x | -0.39024 | 0.46792 | 0.01983 | 0.12083 | 0.01916 | -0.04909 | -0.00148 | 0.04449 | 0.09356 |
| `MultiPosture local CSV` | `left_eye_outer` | y | -0.64127 | 0.19437 | -0.47706 | 0.11884 | -0.51268 | -0.56986 | -0.54356 | -0.46936 | -0.33839 |
| `MultiPosture local CSV` | `left_eye_outer` | z | -0.79561 | 0.10372 | -0.32897 | 0.15610 | -0.32591 | -0.55172 | -0.41488 | -0.23023 | -0.14135 |
| `MultiPosture local CSV` | `right_eye_inner` | x | -0.41367 | 0.45349 | -0.00405 | 0.12271 | -0.00573 | -0.07122 | -0.02607 | 0.02017 | 0.07095 |
| `MultiPosture local CSV` | `right_eye_inner` | y | -0.64398 | 0.21192 | -0.48172 | 0.11954 | -0.51768 | -0.57478 | -0.54918 | -0.47327 | -0.34494 |
| `MultiPosture local CSV` | `right_eye_inner` | z | -0.80257 | 0.09239 | -0.34895 | 0.15598 | -0.34626 | -0.57289 | -0.43443 | -0.25067 | -0.16108 |
| `MultiPosture local CSV` | `right_eye` | x | -0.41015 | 0.45222 | -0.00120 | 0.12207 | -0.00269 | -0.06837 | -0.02284 | 0.02327 | 0.07367 |
| `MultiPosture local CSV` | `right_eye` | y | -0.64489 | 0.20700 | -0.48034 | 0.11953 | -0.51609 | -0.57377 | -0.54797 | -0.47179 | -0.34211 |
| `MultiPosture local CSV` | `right_eye` | z | -0.82791 | 0.08113 | -0.36136 | 0.15714 | -0.35842 | -0.58793 | -0.44723 | -0.26147 | -0.17264 |
| `MultiPosture local CSV` | `right_eye_outer` | x | -0.40225 | 0.44757 | 0.00326 | 0.11997 | 0.00208 | -0.06240 | -0.01776 | 0.02762 | 0.07641 |
| `MultiPosture local CSV` | `right_eye_outer` | y | -0.63544 | 0.19879 | -0.47115 | 0.11886 | -0.50679 | -0.56440 | -0.53853 | -0.46213 | -0.33395 |
| `MultiPosture local CSV` | `right_eye_outer` | z | -0.79755 | 0.09762 | -0.34067 | 0.15573 | -0.33827 | -0.56487 | -0.42610 | -0.24209 | -0.15348 |
| `MultiPosture local CSV` | `left_ear` | x | -0.34276 | 0.49815 | 0.07977 | 0.11687 | 0.08088 | 0.03221 | 0.06240 | 0.09931 | 0.14987 |
| `MultiPosture local CSV` | `left_ear` | y | -0.60649 | 0.14393 | -0.48168 | 0.10339 | -0.51444 | -0.55926 | -0.53803 | -0.47984 | -0.33771 |
| `MultiPosture local CSV` | `left_ear` | z | -0.72739 | 0.19111 | -0.22797 | 0.15982 | -0.22454 | -0.45090 | -0.31426 | -0.12446 | -0.03587 |
| `MultiPosture local CSV` | `right_ear` | x | -0.42536 | 0.35447 | -0.06053 | 0.10702 | -0.06160 | -0.11341 | -0.08137 | -0.04370 | 0.00083 |
| `MultiPosture local CSV` | `right_ear` | y | -0.55684 | 0.11099 | -0.43742 | 0.10308 | -0.47224 | -0.51535 | -0.49445 | -0.43094 | -0.31925 |
| `MultiPosture local CSV` | `right_ear` | z | -0.72641 | 0.19395 | -0.22960 | 0.16058 | -0.22633 | -0.45780 | -0.31601 | -0.12562 | -0.03590 |
| `MultiPosture local CSV` | `mouth_left` | x | -0.35145 | 0.46712 | 0.04120 | 0.11784 | 0.04096 | -0.01937 | 0.02135 | 0.06414 | 0.11042 |
| `MultiPosture local CSV` | `mouth_left` | y | -0.59930 | 0.22090 | -0.44755 | 0.11243 | -0.48034 | -0.53669 | -0.51070 | -0.44139 | -0.30152 |
| `MultiPosture local CSV` | `mouth_left` | z | -0.74005 | 0.11988 | -0.29166 | 0.14843 | -0.28901 | -0.50017 | -0.37520 | -0.19642 | -0.10942 |
| `MultiPosture local CSV` | `mouth_right` | x | -0.35820 | 0.40585 | 0.00635 | 0.10984 | 0.00559 | -0.05054 | -0.01315 | 0.02749 | 0.07295 |
| `MultiPosture local CSV` | `mouth_right` | y | -0.56362 | 0.19413 | -0.41516 | 0.10980 | -0.44710 | -0.50280 | -0.47738 | -0.40637 | -0.28307 |
| `MultiPosture local CSV` | `mouth_right` | z | -0.75851 | 0.09749 | -0.31956 | 0.14872 | -0.31696 | -0.53524 | -0.40265 | -0.22402 | -0.13602 |
| `MultiPosture local CSV` | `left_shoulder` | x | -0.21704 | 0.50218 | 0.17980 | 0.09638 | 0.18344 | 0.13490 | 0.16895 | 0.19719 | 0.24475 |
| `MultiPosture local CSV` | `left_shoulder` | y | -0.52287 | 0.16947 | -0.35686 | 0.07355 | -0.38104 | -0.41311 | -0.40082 | -0.34544 | -0.25244 |
| `MultiPosture local CSV` | `left_shoulder` | z | -0.47896 | 0.24370 | -0.09456 | 0.13208 | -0.09305 | -0.28398 | -0.16889 | -0.00773 | 0.06710 |
| `MultiPosture local CSV` | `right_shoulder` | x | -0.52177 | 0.26671 | -0.13802 | 0.10905 | -0.14249 | -0.18245 | -0.15784 | -0.12515 | -0.07896 |
| `MultiPosture local CSV` | `right_shoulder` | y | -0.51274 | 0.23428 | -0.40694 | 0.07836 | -0.43287 | -0.46718 | -0.45367 | -0.38909 | -0.30757 |
| `MultiPosture local CSV` | `right_shoulder` | z | -0.43102 | 0.28425 | -0.04524 | 0.12988 | -0.03606 | -0.24580 | -0.11405 | 0.04206 | 0.10966 |
| `MultiPosture local CSV` | `left_elbow` | x | -0.07073 | 0.71585 | 0.24443 | 0.10199 | 0.24542 | 0.15708 | 0.19601 | 0.28319 | 0.33117 |
| `MultiPosture local CSV` | `left_elbow` | y | -0.69051 | 0.28725 | -0.16521 | 0.07888 | -0.17283 | -0.23603 | -0.20680 | -0.14502 | -0.08283 |
| `MultiPosture local CSV` | `left_elbow` | z | -0.51049 | 0.28556 | -0.11785 | 0.11367 | -0.11542 | -0.28212 | -0.17728 | -0.04277 | 0.02309 |
| `MultiPosture local CSV` | `right_elbow` | x | -0.64366 | 0.13951 | -0.19185 | 0.09560 | -0.19858 | -0.26213 | -0.22377 | -0.15428 | -0.08701 |
| `MultiPosture local CSV` | `right_elbow` | y | -0.55488 | 0.29722 | -0.14106 | 0.07747 | -0.15876 | -0.20967 | -0.18368 | -0.12032 | -0.05367 |
| `MultiPosture local CSV` | `right_elbow` | z | -0.53331 | 0.21822 | -0.09903 | 0.12028 | -0.08865 | -0.28183 | -0.16698 | -0.01515 | 0.04389 |
| `MultiPosture local CSV` | `left_wrist` | x | -0.23691 | 0.67923 | 0.17912 | 0.14463 | 0.13961 | 0.02700 | 0.09552 | 0.26891 | 0.36085 |
| `MultiPosture local CSV` | `left_wrist` | y | -0.66072 | 0.41859 | -0.04083 | 0.12579 | -0.03860 | -0.19022 | -0.10046 | 0.02221 | 0.08855 |
| `MultiPosture local CSV` | `left_wrist` | z | -0.48980 | 0.52635 | -0.15269 | 0.11837 | -0.16475 | -0.29721 | -0.22667 | -0.08199 | -0.00690 |
| `MultiPosture local CSV` | `right_wrist` | x | -0.61740 | 0.29065 | -0.15249 | 0.13628 | -0.12452 | -0.31725 | -0.25418 | -0.08136 | 0.01926 |
| `MultiPosture local CSV` | `right_wrist` | y | -0.70332 | 0.50476 | -0.04004 | 0.12858 | -0.04117 | -0.18012 | -0.09032 | 0.02146 | 0.08739 |
| `MultiPosture local CSV` | `right_wrist` | z | -0.70376 | 0.32983 | -0.21508 | 0.12591 | -0.21847 | -0.38296 | -0.29171 | -0.12778 | -0.06222 |
| `MultiPosture local CSV` | `left_pinky` | x | -0.30057 | 0.65875 | 0.15708 | 0.16471 | 0.10582 | -0.01884 | 0.06085 | 0.27047 | 0.36984 |
| `MultiPosture local CSV` | `left_pinky` | y | -0.65040 | 0.51994 | -0.02115 | 0.14621 | -0.01606 | -0.19621 | -0.08345 | 0.06000 | 0.12466 |
| `MultiPosture local CSV` | `left_pinky` | z | -0.50976 | 0.55568 | -0.16354 | 0.12317 | -0.17624 | -0.31051 | -0.24613 | -0.09154 | -0.01241 |
| `MultiPosture local CSV` | `right_pinky` | x | -0.63441 | 0.36109 | -0.11170 | 0.15513 | -0.07389 | -0.31339 | -0.23526 | -0.02375 | 0.05084 |
| `MultiPosture local CSV` | `right_pinky` | y | -0.74250 | 0.56426 | -0.02323 | 0.15059 | -0.01789 | -0.19885 | -0.07590 | 0.05788 | 0.12596 |
| `MultiPosture local CSV` | `right_pinky` | z | -0.77783 | 0.29415 | -0.27124 | 0.13379 | -0.27699 | -0.45122 | -0.35094 | -0.18177 | -0.10865 |
| `MultiPosture local CSV` | `left_index` | x | -0.32758 | 0.63641 | 0.11666 | 0.16915 | 0.05784 | -0.05941 | 0.01369 | 0.22789 | 0.35112 |
| `MultiPosture local CSV` | `left_index` | y | -0.64019 | 0.52218 | -0.03702 | 0.15327 | -0.02994 | -0.21165 | -0.11759 | 0.05858 | 0.12035 |
| `MultiPosture local CSV` | `left_index` | z | -0.52139 | 0.57280 | -0.17488 | 0.12652 | -0.19014 | -0.32622 | -0.25972 | -0.10022 | -0.01846 |
| `MultiPosture local CSV` | `right_index` | x | -0.61543 | 0.41940 | -0.07321 | 0.16388 | -0.02818 | -0.29203 | -0.20026 | 0.02024 | 0.09623 |
| `MultiPosture local CSV` | `right_index` | y | -0.79239 | 0.59278 | -0.07337 | 0.15978 | -0.05868 | -0.27514 | -0.13611 | 0.01362 | 0.08117 |
| `MultiPosture local CSV` | `right_index` | z | -0.77396 | 0.30670 | -0.28068 | 0.13255 | -0.29037 | -0.45396 | -0.36312 | -0.18977 | -0.11857 |
| `MultiPosture local CSV` | `left_thumb` | x | -0.24610 | 0.65318 | 0.16262 | 0.14751 | 0.12188 | 0.00830 | 0.07591 | 0.25675 | 0.35640 |
| `MultiPosture local CSV` | `left_thumb` | y | -0.63599 | 0.44264 | -0.02399 | 0.13276 | -0.02521 | -0.18390 | -0.08818 | 0.04778 | 0.11382 |
| `MultiPosture local CSV` | `left_thumb` | z | -0.49221 | 0.53250 | -0.15826 | 0.11931 | -0.17161 | -0.30317 | -0.23710 | -0.08546 | -0.01075 |
| `MultiPosture local CSV` | `right_thumb` | x | -0.61111 | 0.31945 | -0.13136 | 0.14249 | -0.09958 | -0.30936 | -0.23540 | -0.05373 | 0.02726 |
| `MultiPosture local CSV` | `right_thumb` | y | -0.72538 | 0.51675 | -0.04955 | 0.13596 | -0.04807 | -0.20309 | -0.10496 | 0.02198 | 0.08525 |
| `MultiPosture local CSV` | `right_thumb` | z | -0.72019 | 0.32115 | -0.22986 | 0.12664 | -0.23600 | -0.39762 | -0.30841 | -0.14181 | -0.07420 |
| `MultiPosture local CSV` | `left_hip` | x | 0.06491 | 0.12921 | 0.10002 | 0.00901 | 0.10171 | 0.08745 | 0.09555 | 0.10622 | 0.10907 |
| `MultiPosture local CSV` | `left_hip` | y | -0.05884 | 0.06082 | 0.00221 | 0.01922 | 0.00305 | -0.01681 | -0.00277 | 0.00699 | 0.01799 |
| `MultiPosture local CSV` | `left_hip` | z | -0.00457 | 0.07053 | 0.02958 | 0.01008 | 0.02942 | 0.01880 | 0.02377 | 0.03487 | 0.04103 |
| `MultiPosture local CSV` | `right_hip` | x | -0.12970 | -0.06575 | -0.09848 | 0.00894 | -0.10027 | -0.10784 | -0.10479 | -0.09394 | -0.08624 |
| `MultiPosture local CSV` | `right_hip` | y | -0.06443 | 0.05289 | -0.00544 | 0.01896 | -0.00598 | -0.02455 | -0.01028 | -0.00042 | 0.01361 |
| `MultiPosture local CSV` | `right_hip` | z | -0.06792 | 0.00832 | -0.02670 | 0.01046 | -0.02660 | -0.03878 | -0.03195 | -0.02095 | -0.01535 |
| `MultiPosture local CSV` | `left_knee` | x | -0.05341 | 0.42232 | 0.18987 | 0.10521 | 0.20136 | 0.05186 | 0.10117 | 0.27867 | 0.32476 |
| `MultiPosture local CSV` | `left_knee` | y | -0.30475 | 0.41899 | 0.21420 | 0.11975 | 0.24646 | 0.05886 | 0.15381 | 0.29717 | 0.33529 |
| `MultiPosture local CSV` | `left_knee` | z | -0.41469 | 0.15745 | -0.26303 | 0.07493 | -0.27422 | -0.35137 | -0.31367 | -0.22357 | -0.16640 |
| `MultiPosture local CSV` | `right_knee` | x | -0.37960 | 0.20763 | -0.14452 | 0.11916 | -0.16467 | -0.29159 | -0.24051 | -0.04911 | 0.02656 |
| `MultiPosture local CSV` | `right_knee` | y | -0.34242 | 0.42795 | 0.13495 | 0.11450 | 0.16027 | -0.03373 | 0.07233 | 0.21708 | 0.24825 |
| `MultiPosture local CSV` | `right_knee` | z | -0.38772 | 0.09830 | -0.26092 | 0.07051 | -0.26858 | -0.34122 | -0.31874 | -0.22089 | -0.17632 |
| `MultiPosture local CSV` | `left_ankle` | x | -0.29518 | 0.60033 | 0.13051 | 0.18467 | 0.14795 | -0.15870 | 0.03903 | 0.25800 | 0.33583 |
| `MultiPosture local CSV` | `left_ankle` | y | -0.09810 | 0.77130 | 0.53542 | 0.15040 | 0.57140 | 0.34377 | 0.50847 | 0.62231 | 0.66473 |
| `MultiPosture local CSV` | `left_ankle` | z | -0.39744 | 0.39788 | -0.10998 | 0.13118 | -0.09148 | -0.30433 | -0.20739 | -0.01334 | 0.04381 |
| `MultiPosture local CSV` | `right_ankle` | x | -0.62480 | 0.33431 | -0.12396 | 0.19006 | -0.13951 | -0.33379 | -0.27074 | -0.00176 | 0.14566 |
| `MultiPosture local CSV` | `right_ankle` | y | -0.08648 | 0.73472 | 0.51327 | 0.13016 | 0.53940 | 0.35351 | 0.48327 | 0.59113 | 0.63003 |
| `MultiPosture local CSV` | `right_ankle` | z | -0.40088 | 0.37641 | -0.10407 | 0.12315 | -0.08194 | -0.27563 | -0.21315 | -0.01844 | 0.04676 |
| `MultiPosture local CSV` | `left_heel` | x | -0.35821 | 0.60853 | 0.10433 | 0.19759 | 0.12585 | -0.21528 | 0.01369 | 0.23889 | 0.32023 |
| `MultiPosture local CSV` | `left_heel` | y | -0.09094 | 0.81410 | 0.56562 | 0.15768 | 0.60466 | 0.38056 | 0.54217 | 0.65692 | 0.69863 |
| `MultiPosture local CSV` | `left_heel` | z | -0.39563 | 0.42226 | -0.08721 | 0.13826 | -0.06873 | -0.29074 | -0.18868 | 0.01688 | 0.07643 |
| `MultiPosture local CSV` | `right_heel` | x | -0.66015 | 0.35803 | -0.12350 | 0.20026 | -0.14258 | -0.33863 | -0.27495 | -0.00647 | 0.18067 |
| `MultiPosture local CSV` | `right_heel` | y | -0.07547 | 0.77307 | 0.54470 | 0.13540 | 0.57259 | 0.37984 | 0.51481 | 0.62383 | 0.66427 |
| `MultiPosture local CSV` | `right_heel` | z | -0.44866 | 0.37005 | -0.12458 | 0.13070 | -0.10280 | -0.30402 | -0.24074 | -0.03335 | 0.03612 |
| `MultiPosture local CSV` | `left_foot_index` | x | -0.45955 | 0.61871 | 0.09995 | 0.23575 | 0.13097 | -0.27548 | -0.01188 | 0.26696 | 0.36917 |
| `MultiPosture local CSV` | `left_foot_index` | y | -0.12525 | 0.86560 | 0.62746 | 0.16254 | 0.66980 | 0.40620 | 0.60692 | 0.71955 | 0.76497 |
| `MultiPosture local CSV` | `left_foot_index` | z | -0.48145 | 0.41206 | -0.14816 | 0.14829 | -0.13582 | -0.36174 | -0.25743 | -0.03882 | 0.02664 |
| `MultiPosture local CSV` | `right_foot_index` | x | -0.73632 | 0.39384 | -0.13701 | 0.23774 | -0.16683 | -0.41959 | -0.32320 | 0.00137 | 0.24000 |
| `MultiPosture local CSV` | `right_foot_index` | y | -0.10139 | 0.81639 | 0.58679 | 0.13936 | 0.61856 | 0.42396 | 0.55794 | 0.66727 | 0.70889 |
| `MultiPosture local CSV` | `right_foot_index` | z | -0.53698 | 0.31847 | -0.20413 | 0.14070 | -0.18064 | -0.39778 | -0.33089 | -0.10256 | -0.02984 |
| `DriveNet MediaPipe normalized image landmarks` | `nose` | x | 0.41930 | 0.59829 | 0.49671 | 0.05419 | 0.48918 | 0.43476 | 0.44250 | 0.56037 | 0.57216 |
| `DriveNet MediaPipe normalized image landmarks` | `nose` | y | 0.21483 | 0.28173 | 0.25390 | 0.01529 | 0.25540 | 0.23224 | 0.24264 | 0.26546 | 0.27542 |
| `DriveNet MediaPipe normalized image landmarks` | `nose` | z | -0.21978 | 0.06526 | -0.05209 | 0.04446 | -0.06356 | -0.09903 | -0.08330 | -0.01921 | 0.01293 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye_inner` | x | 0.43505 | 0.60762 | 0.50919 | 0.05310 | 0.49943 | 0.44808 | 0.45892 | 0.57189 | 0.58398 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye_inner` | y | 0.19659 | 0.25670 | 0.22739 | 0.01447 | 0.22456 | 0.20960 | 0.21609 | 0.23877 | 0.25030 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye_inner` | z | -0.24232 | 0.04399 | -0.08463 | 0.04753 | -0.09779 | -0.13452 | -0.11821 | -0.05206 | -0.01184 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye` | x | 0.43951 | 0.61402 | 0.51377 | 0.05361 | 0.50329 | 0.45239 | 0.46348 | 0.57718 | 0.58967 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye` | y | 0.19688 | 0.25562 | 0.22668 | 0.01420 | 0.22383 | 0.20921 | 0.21561 | 0.23785 | 0.24905 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye` | z | -0.24256 | 0.04374 | -0.08491 | 0.04755 | -0.09809 | -0.13477 | -0.11853 | -0.05232 | -0.01211 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye_outer` | x | 0.44377 | 0.61975 | 0.51808 | 0.05406 | 0.50699 | 0.45647 | 0.46774 | 0.58198 | 0.59504 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye_outer` | y | 0.19682 | 0.25449 | 0.22612 | 0.01395 | 0.22339 | 0.20848 | 0.21538 | 0.23724 | 0.24754 |
| `DriveNet MediaPipe normalized image landmarks` | `left_eye_outer` | z | -0.24263 | 0.04390 | -0.08479 | 0.04758 | -0.09802 | -0.13464 | -0.11844 | -0.05216 | -0.01195 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye_inner` | x | 0.43439 | 0.59198 | 0.50603 | 0.05077 | 0.49738 | 0.44638 | 0.45773 | 0.56788 | 0.57649 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye_inner` | y | 0.19765 | 0.25650 | 0.22770 | 0.01410 | 0.22448 | 0.21097 | 0.21606 | 0.23839 | 0.24962 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye_inner` | z | -0.20352 | 0.08578 | -0.03988 | 0.04655 | -0.05214 | -0.08863 | -0.07282 | -0.00724 | 0.03108 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye` | x | 0.43798 | 0.58923 | 0.50813 | 0.04971 | 0.49955 | 0.44989 | 0.46080 | 0.56967 | 0.57633 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye` | y | 0.19882 | 0.25540 | 0.22735 | 0.01361 | 0.22391 | 0.21192 | 0.21592 | 0.23735 | 0.24870 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye` | z | -0.20395 | 0.08525 | -0.04042 | 0.04654 | -0.05261 | -0.08910 | -0.07336 | -0.00776 | 0.03057 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye_outer` | x | 0.44154 | 0.58672 | 0.51033 | 0.04863 | 0.50175 | 0.45319 | 0.46383 | 0.57153 | 0.57671 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye_outer` | y | 0.19959 | 0.25396 | 0.22687 | 0.01310 | 0.22312 | 0.21249 | 0.21614 | 0.23648 | 0.24743 |
| `DriveNet MediaPipe normalized image landmarks` | `right_eye_outer` | z | -0.20423 | 0.08494 | -0.04069 | 0.04653 | -0.05286 | -0.08932 | -0.07363 | -0.00803 | 0.03033 |
| `DriveNet MediaPipe normalized image landmarks` | `left_ear` | x | 0.48013 | 0.64041 | 0.54839 | 0.05117 | 0.53560 | 0.49105 | 0.50238 | 0.61081 | 0.62117 |
| `DriveNet MediaPipe normalized image landmarks` | `left_ear` | y | 0.21123 | 0.25768 | 0.23542 | 0.01136 | 0.23479 | 0.22052 | 0.22629 | 0.24519 | 0.25089 |
| `DriveNet MediaPipe normalized image landmarks` | `left_ear` | z | -0.28462 | -0.00280 | -0.16547 | 0.06039 | -0.18540 | -0.22361 | -0.20763 | -0.14417 | -0.05517 |
| `DriveNet MediaPipe normalized image landmarks` | `right_ear` | x | 0.47246 | 0.60888 | 0.53962 | 0.04527 | 0.53050 | 0.48716 | 0.49724 | 0.59765 | 0.60091 |
| `DriveNet MediaPipe normalized image landmarks` | `right_ear` | y | 0.21377 | 0.25782 | 0.23645 | 0.01050 | 0.23376 | 0.22450 | 0.22896 | 0.24459 | 0.25201 |
| `DriveNet MediaPipe normalized image landmarks` | `right_ear` | z | -0.11107 | 0.18882 | 0.03721 | 0.05642 | 0.02264 | -0.02161 | -0.00378 | 0.06297 | 0.13318 |
| `DriveNet MediaPipe normalized image landmarks` | `mouth_left` | x | 0.43878 | 0.61443 | 0.51308 | 0.05307 | 0.50105 | 0.45453 | 0.46282 | 0.57530 | 0.58844 |
| `DriveNet MediaPipe normalized image landmarks` | `mouth_left` | y | 0.24680 | 0.30320 | 0.28021 | 0.01280 | 0.28262 | 0.26176 | 0.27103 | 0.28846 | 0.29724 |
| `DriveNet MediaPipe normalized image landmarks` | `mouth_left` | z | -0.22923 | 0.05511 | -0.07251 | 0.04770 | -0.08652 | -0.12253 | -0.10677 | -0.04054 | 0.00228 |
| `DriveNet MediaPipe normalized image landmarks` | `mouth_right` | x | 0.43608 | 0.59906 | 0.50869 | 0.05088 | 0.49846 | 0.45176 | 0.45943 | 0.57042 | 0.57958 |
| `DriveNet MediaPipe normalized image landmarks` | `mouth_right` | y | 0.24870 | 0.30324 | 0.28045 | 0.01221 | 0.28091 | 0.26363 | 0.27240 | 0.28920 | 0.29736 |
| `DriveNet MediaPipe normalized image landmarks` | `mouth_right` | z | -0.18100 | 0.10717 | -0.01682 | 0.04643 | -0.02945 | -0.06528 | -0.04967 | 0.01730 | 0.05483 |
| `DriveNet MediaPipe normalized image landmarks` | `left_shoulder` | x | 0.53678 | 0.69853 | 0.61278 | 0.05171 | 0.60920 | 0.54556 | 0.56308 | 0.66817 | 0.68863 |
| `DriveNet MediaPipe normalized image landmarks` | `left_shoulder` | y | 0.35446 | 0.39242 | 0.37217 | 0.01026 | 0.37085 | 0.35871 | 0.36362 | 0.38150 | 0.38758 |
| `DriveNet MediaPipe normalized image landmarks` | `left_shoulder` | z | -0.36059 | -0.09053 | -0.24998 | 0.05750 | -0.27148 | -0.30160 | -0.28797 | -0.23558 | -0.14452 |
| `DriveNet MediaPipe normalized image landmarks` | `right_shoulder` | x | 0.49098 | 0.59416 | 0.54524 | 0.02886 | 0.54507 | 0.50374 | 0.51892 | 0.57371 | 0.58185 |
| `DriveNet MediaPipe normalized image landmarks` | `right_shoulder` | y | 0.33225 | 0.38712 | 0.35861 | 0.01413 | 0.35903 | 0.33706 | 0.34873 | 0.37025 | 0.37779 |
| `DriveNet MediaPipe normalized image landmarks` | `right_shoulder` | z | 0.03232 | 0.29801 | 0.17268 | 0.04870 | 0.16092 | 0.12021 | 0.13910 | 0.19287 | 0.25770 |
| `DriveNet MediaPipe normalized image landmarks` | `left_elbow` | x | 0.47432 | 0.60276 | 0.55108 | 0.03057 | 0.54711 | 0.51074 | 0.53336 | 0.57752 | 0.58689 |
| `DriveNet MediaPipe normalized image landmarks` | `left_elbow` | y | 0.48943 | 0.62920 | 0.57136 | 0.04415 | 0.59300 | 0.49805 | 0.53170 | 0.60862 | 0.61522 |
| `DriveNet MediaPipe normalized image landmarks` | `left_elbow` | z | -0.45745 | -0.15469 | -0.31233 | 0.06320 | -0.30963 | -0.39989 | -0.35999 | -0.27444 | -0.22108 |
| `DriveNet MediaPipe normalized image landmarks` | `right_elbow` | x | 0.42034 | 0.54307 | 0.47234 | 0.03270 | 0.45716 | 0.43964 | 0.44760 | 0.50454 | 0.52291 |
| `DriveNet MediaPipe normalized image landmarks` | `right_elbow` | y | 0.38953 | 0.55572 | 0.47633 | 0.05050 | 0.49585 | 0.40311 | 0.41323 | 0.51775 | 0.53535 |
| `DriveNet MediaPipe normalized image landmarks` | `right_elbow` | z | 0.06713 | 0.37022 | 0.25053 | 0.05184 | 0.25525 | 0.18447 | 0.21339 | 0.28961 | 0.31287 |
| `DriveNet MediaPipe normalized image landmarks` | `left_wrist` | x | 0.31355 | 0.46192 | 0.39049 | 0.03364 | 0.39124 | 0.35051 | 0.36370 | 0.41749 | 0.43227 |
| `DriveNet MediaPipe normalized image landmarks` | `left_wrist` | y | 0.37100 | 0.56888 | 0.51356 | 0.04555 | 0.53824 | 0.45375 | 0.47618 | 0.54755 | 0.55466 |
| `DriveNet MediaPipe normalized image landmarks` | `left_wrist` | z | -0.51238 | -0.08466 | -0.27185 | 0.08521 | -0.25692 | -0.39875 | -0.31967 | -0.20854 | -0.17040 |
| `DriveNet MediaPipe normalized image landmarks` | `right_wrist` | x | 0.30317 | 0.44810 | 0.36458 | 0.04275 | 0.34599 | 0.31950 | 0.33142 | 0.41370 | 0.42776 |
| `DriveNet MediaPipe normalized image landmarks` | `right_wrist` | y | 0.33147 | 0.57777 | 0.42473 | 0.06087 | 0.41977 | 0.35127 | 0.36938 | 0.47060 | 0.52083 |
| `DriveNet MediaPipe normalized image landmarks` | `right_wrist` | z | -0.18958 | 0.42299 | 0.17247 | 0.11160 | 0.16351 | 0.03172 | 0.11343 | 0.25496 | 0.32783 |
| `DriveNet MediaPipe normalized image landmarks` | `left_pinky` | x | 0.26524 | 0.42164 | 0.34772 | 0.03472 | 0.35112 | 0.30714 | 0.31873 | 0.37713 | 0.39347 |
| `DriveNet MediaPipe normalized image landmarks` | `left_pinky` | y | 0.36064 | 0.58290 | 0.51418 | 0.04974 | 0.53829 | 0.45502 | 0.47151 | 0.55173 | 0.56250 |
| `DriveNet MediaPipe normalized image landmarks` | `left_pinky` | z | -0.55577 | -0.09063 | -0.29845 | 0.09188 | -0.28898 | -0.42989 | -0.35163 | -0.22751 | -0.18443 |
| `DriveNet MediaPipe normalized image landmarks` | `right_pinky` | x | 0.26867 | 0.42230 | 0.33565 | 0.04527 | 0.31412 | 0.28820 | 0.29948 | 0.38945 | 0.40000 |
| `DriveNet MediaPipe normalized image landmarks` | `right_pinky` | y | 0.30201 | 0.59230 | 0.40652 | 0.07150 | 0.38989 | 0.32347 | 0.34632 | 0.46136 | 0.52600 |
| `DriveNet MediaPipe normalized image landmarks` | `right_pinky` | z | -0.23328 | 0.43972 | 0.16094 | 0.12315 | 0.14776 | 0.00801 | 0.09535 | 0.25497 | 0.33488 |
| `DriveNet MediaPipe normalized image landmarks` | `left_index` | x | 0.26323 | 0.41352 | 0.34594 | 0.03588 | 0.34890 | 0.30194 | 0.31727 | 0.37590 | 0.39484 |
| `DriveNet MediaPipe normalized image landmarks` | `left_index` | y | 0.33338 | 0.54108 | 0.47753 | 0.04930 | 0.50091 | 0.41640 | 0.43618 | 0.51482 | 0.52441 |
| `DriveNet MediaPipe normalized image landmarks` | `left_index` | z | -0.53193 | -0.07994 | -0.28787 | 0.08951 | -0.28299 | -0.41466 | -0.34187 | -0.21567 | -0.17215 |
| `DriveNet MediaPipe normalized image landmarks` | `right_index` | x | 0.26841 | 0.42051 | 0.33157 | 0.04499 | 0.31078 | 0.28590 | 0.29570 | 0.38593 | 0.39572 |
| `DriveNet MediaPipe normalized image landmarks` | `right_index` | y | 0.28474 | 0.59016 | 0.39443 | 0.06806 | 0.38263 | 0.31517 | 0.33883 | 0.43472 | 0.51041 |
| `DriveNet MediaPipe normalized image landmarks` | `right_index` | z | -0.25065 | 0.41421 | 0.13262 | 0.12510 | 0.11190 | -0.01882 | 0.05889 | 0.23290 | 0.31144 |
| `DriveNet MediaPipe normalized image landmarks` | `left_thumb` | x | 0.27698 | 0.43006 | 0.35938 | 0.03529 | 0.36089 | 0.31631 | 0.33279 | 0.38684 | 0.40977 |
| `DriveNet MediaPipe normalized image landmarks` | `left_thumb` | y | 0.33698 | 0.54217 | 0.47632 | 0.04711 | 0.49739 | 0.41578 | 0.43943 | 0.51101 | 0.52149 |
| `DriveNet MediaPipe normalized image landmarks` | `left_thumb` | z | -0.49918 | -0.07142 | -0.26218 | 0.08510 | -0.25140 | -0.38753 | -0.31025 | -0.19856 | -0.15837 |
| `DriveNet MediaPipe normalized image landmarks` | `right_thumb` | x | 0.27745 | 0.42342 | 0.34247 | 0.04457 | 0.32299 | 0.29560 | 0.30880 | 0.39657 | 0.40630 |
| `DriveNet MediaPipe normalized image landmarks` | `right_thumb` | y | 0.31191 | 0.58500 | 0.40474 | 0.06212 | 0.39753 | 0.33340 | 0.35035 | 0.44419 | 0.51209 |
| `DriveNet MediaPipe normalized image landmarks` | `right_thumb` | z | -0.20724 | 0.41175 | 0.15498 | 0.11421 | 0.14316 | 0.01241 | 0.09145 | 0.24206 | 0.31595 |
| `DriveNet MediaPipe normalized image landmarks` | `left_hip` | x | 0.49463 | 0.61314 | 0.55442 | 0.03596 | 0.56172 | 0.50209 | 0.51445 | 0.58754 | 0.59360 |
| `DriveNet MediaPipe normalized image landmarks` | `left_hip` | y | 0.62121 | 0.73349 | 0.68354 | 0.03162 | 0.69899 | 0.63114 | 0.65149 | 0.70716 | 0.71347 |
| `DriveNet MediaPipe normalized image landmarks` | `left_hip` | z | -0.15077 | -0.11523 | -0.13089 | 0.00711 | -0.12979 | -0.14067 | -0.13658 | -0.12602 | -0.12226 |
| `DriveNet MediaPipe normalized image landmarks` | `right_hip` | x | 0.45344 | 0.58475 | 0.51485 | 0.03696 | 0.51940 | 0.46489 | 0.47069 | 0.54740 | 0.56044 |
| `DriveNet MediaPipe normalized image landmarks` | `right_hip` | y | 0.55506 | 0.70370 | 0.62318 | 0.03592 | 0.64041 | 0.56527 | 0.58385 | 0.64801 | 0.65286 |
| `DriveNet MediaPipe normalized image landmarks` | `right_hip` | z | 0.11506 | 0.15055 | 0.13064 | 0.00714 | 0.12954 | 0.12200 | 0.12570 | 0.13632 | 0.14049 |
| `DriveNet MediaPipe normalized image landmarks` | `left_knee` | x | 0.25483 | 0.38380 | 0.31987 | 0.04323 | 0.33907 | 0.25877 | 0.26178 | 0.35529 | 0.36167 |
| `DriveNet MediaPipe normalized image landmarks` | `left_knee` | y | 0.59986 | 0.74642 | 0.65281 | 0.04291 | 0.63561 | 0.60674 | 0.61653 | 0.70326 | 0.71525 |
| `DriveNet MediaPipe normalized image landmarks` | `left_knee` | z | -0.26617 | -0.04733 | -0.16784 | 0.04353 | -0.17148 | -0.22694 | -0.19855 | -0.13182 | -0.11226 |
| `DriveNet MediaPipe normalized image landmarks` | `right_knee` | x | 0.24795 | 0.39049 | 0.32416 | 0.04434 | 0.34439 | 0.25701 | 0.27009 | 0.35876 | 0.37002 |
| `DriveNet MediaPipe normalized image landmarks` | `right_knee` | y | 0.49773 | 0.69936 | 0.56815 | 0.03726 | 0.58406 | 0.51195 | 0.52740 | 0.59768 | 0.60678 |
| `DriveNet MediaPipe normalized image landmarks` | `right_knee` | z | -0.01171 | 0.30027 | 0.11582 | 0.06416 | 0.10407 | 0.03964 | 0.06366 | 0.16922 | 0.20821 |
| `DriveNet MediaPipe normalized image landmarks` | `left_ankle` | x | 0.12242 | 0.42078 | 0.21834 | 0.05946 | 0.23329 | 0.14177 | 0.14977 | 0.26532 | 0.27609 |
| `DriveNet MediaPipe normalized image landmarks` | `left_ankle` | y | 0.70421 | 0.94857 | 0.87621 | 0.03443 | 0.88107 | 0.83276 | 0.85223 | 0.90160 | 0.91486 |
| `DriveNet MediaPipe normalized image landmarks` | `left_ankle` | z | -0.18149 | 0.32090 | 0.01578 | 0.08449 | 0.00948 | -0.08928 | -0.04125 | 0.06786 | 0.12114 |
| `DriveNet MediaPipe normalized image landmarks` | `right_ankle` | x | 0.11697 | 0.44896 | 0.22469 | 0.05910 | 0.24381 | 0.13573 | 0.15949 | 0.26943 | 0.28918 |
| `DriveNet MediaPipe normalized image landmarks` | `right_ankle` | y | 0.72075 | 0.88534 | 0.80102 | 0.03362 | 0.79876 | 0.76015 | 0.77685 | 0.81966 | 0.85126 |
| `DriveNet MediaPipe normalized image landmarks` | `right_ankle` | z | 0.03832 | 0.39404 | 0.17112 | 0.06403 | 0.16542 | 0.09316 | 0.12058 | 0.21173 | 0.25507 |
| `DriveNet MediaPipe normalized image landmarks` | `left_heel` | x | 0.13451 | 0.45733 | 0.23256 | 0.06179 | 0.24304 | 0.15519 | 0.16506 | 0.28020 | 0.29574 |
| `DriveNet MediaPipe normalized image landmarks` | `left_heel` | y | 0.71157 | 0.98333 | 0.92010 | 0.03469 | 0.92746 | 0.87746 | 0.89899 | 0.94563 | 0.95663 |
| `DriveNet MediaPipe normalized image landmarks` | `left_heel` | z | -0.17588 | 0.37252 | 0.03706 | 0.09117 | 0.03038 | -0.07325 | -0.02530 | 0.09172 | 0.14956 |
| `DriveNet MediaPipe normalized image landmarks` | `right_heel` | x | 0.12003 | 0.48303 | 0.23401 | 0.05888 | 0.25344 | 0.14404 | 0.17406 | 0.27915 | 0.29919 |
| `DriveNet MediaPipe normalized image landmarks` | `right_heel` | y | 0.77040 | 0.92733 | 0.85103 | 0.03086 | 0.85079 | 0.80979 | 0.82794 | 0.87035 | 0.89238 |
| `DriveNet MediaPipe normalized image landmarks` | `right_heel` | z | 0.04107 | 0.40742 | 0.17712 | 0.06694 | 0.17158 | 0.09573 | 0.12266 | 0.21855 | 0.26602 |
| `DriveNet MediaPipe normalized image landmarks` | `left_foot_index` | x | 0.00763 | 0.34688 | 0.10980 | 0.06172 | 0.12607 | 0.03046 | 0.04717 | 0.15894 | 0.16979 |
| `DriveNet MediaPipe normalized image landmarks` | `left_foot_index` | y | 0.67651 | 1.01295 | 0.90026 | 0.04814 | 0.89692 | 0.84443 | 0.87251 | 0.93994 | 0.95916 |
| `DriveNet MediaPipe normalized image landmarks` | `left_foot_index` | z | -0.25918 | 0.52192 | 0.03018 | 0.13115 | 0.01176 | -0.12308 | -0.06108 | 0.10954 | 0.19277 |
| `DriveNet MediaPipe normalized image landmarks` | `right_foot_index` | x | 0.02601 | 0.39108 | 0.14388 | 0.06292 | 0.16561 | 0.05168 | 0.07224 | 0.19057 | 0.21081 |
| `DriveNet MediaPipe normalized image landmarks` | `right_foot_index` | y | 0.66681 | 0.94655 | 0.83676 | 0.04661 | 0.83223 | 0.78170 | 0.80605 | 0.86900 | 0.90301 |
| `DriveNet MediaPipe normalized image landmarks` | `right_foot_index` | z | -0.01907 | 0.41310 | 0.15830 | 0.08995 | 0.15221 | 0.04385 | 0.08778 | 0.22092 | 0.28449 |
| `DriveNet MediaPipe world landmarks` | `nose` | x | -0.28681 | 0.15963 | -0.10272 | 0.15024 | -0.16254 | -0.25553 | -0.23722 | 0.08259 | 0.11394 |
| `DriveNet MediaPipe world landmarks` | `nose` | y | -0.55535 | -0.41134 | -0.49926 | 0.03681 | -0.51585 | -0.53847 | -0.52764 | -0.46142 | -0.44326 |
| `DriveNet MediaPipe world landmarks` | `nose` | z | -0.21426 | -0.02553 | -0.10184 | 0.02504 | -0.10554 | -0.12797 | -0.11848 | -0.08477 | -0.06862 |
| `DriveNet MediaPipe world landmarks` | `left_eye_inner` | x | -0.25726 | 0.17788 | -0.08486 | 0.14689 | -0.14640 | -0.23482 | -0.21416 | 0.09406 | 0.12774 |
| `DriveNet MediaPipe world landmarks` | `left_eye_inner` | y | -0.57238 | -0.43607 | -0.51863 | 0.03413 | -0.53360 | -0.55501 | -0.54591 | -0.48430 | -0.46711 |
| `DriveNet MediaPipe world landmarks` | `left_eye_inner` | z | -0.23117 | -0.04616 | -0.13205 | 0.02719 | -0.13753 | -0.15980 | -0.15144 | -0.11469 | -0.09399 |
| `DriveNet MediaPipe world landmarks` | `left_eye` | x | -0.25619 | 0.17991 | -0.08318 | 0.14713 | -0.14459 | -0.23381 | -0.21230 | 0.09635 | 0.12950 |
| `DriveNet MediaPipe world landmarks` | `left_eye` | y | -0.57124 | -0.43418 | -0.51756 | 0.03422 | -0.53295 | -0.55336 | -0.54488 | -0.48310 | -0.46523 |
| `DriveNet MediaPipe world landmarks` | `left_eye` | z | -0.21993 | -0.03530 | -0.12186 | 0.02752 | -0.12761 | -0.14982 | -0.14141 | -0.10524 | -0.08325 |
| `DriveNet MediaPipe world landmarks` | `left_eye_outer` | x | -0.25730 | 0.18015 | -0.08386 | 0.14767 | -0.14532 | -0.23469 | -0.21349 | 0.09627 | 0.12969 |
| `DriveNet MediaPipe world landmarks` | `left_eye_outer` | y | -0.57157 | -0.43417 | -0.51785 | 0.03450 | -0.53335 | -0.55407 | -0.54533 | -0.48294 | -0.46528 |
| `DriveNet MediaPipe world landmarks` | `left_eye_outer` | z | -0.22011 | -0.03933 | -0.12370 | 0.02671 | -0.12894 | -0.15111 | -0.14260 | -0.10674 | -0.08693 |
| `DriveNet MediaPipe world landmarks` | `right_eye_inner` | x | -0.26098 | 0.17247 | -0.07835 | 0.14485 | -0.13509 | -0.22767 | -0.20460 | 0.10039 | 0.13202 |
| `DriveNet MediaPipe world landmarks` | `right_eye_inner` | y | -0.58785 | -0.44201 | -0.53209 | 0.03952 | -0.55230 | -0.56992 | -0.56424 | -0.48801 | -0.47219 |
| `DriveNet MediaPipe world landmarks` | `right_eye_inner` | z | -0.21734 | -0.02808 | -0.11128 | 0.02656 | -0.11606 | -0.13851 | -0.12986 | -0.09479 | -0.07501 |
| `DriveNet MediaPipe world landmarks` | `right_eye` | x | -0.25726 | 0.17239 | -0.07793 | 0.14444 | -0.13472 | -0.22640 | -0.20391 | 0.10032 | 0.13185 |
| `DriveNet MediaPipe world landmarks` | `right_eye` | y | -0.58340 | -0.43790 | -0.52784 | 0.03923 | -0.54790 | -0.56541 | -0.55954 | -0.48439 | -0.46833 |
| `DriveNet MediaPipe world landmarks` | `right_eye` | z | -0.23067 | -0.03958 | -0.12232 | 0.02636 | -0.12663 | -0.14942 | -0.14065 | -0.10623 | -0.08673 |
| `DriveNet MediaPipe world landmarks` | `right_eye_outer` | x | -0.24897 | 0.16796 | -0.07450 | 0.13964 | -0.12924 | -0.21814 | -0.19554 | 0.09817 | 0.12908 |
| `DriveNet MediaPipe world landmarks` | `right_eye_outer` | y | -0.57075 | -0.42503 | -0.51526 | 0.03952 | -0.53559 | -0.55286 | -0.54727 | -0.47149 | -0.45517 |
| `DriveNet MediaPipe world landmarks` | `right_eye_outer` | z | -0.20909 | -0.02145 | -0.10297 | 0.02608 | -0.10716 | -0.12951 | -0.12108 | -0.08674 | -0.06816 |
| `DriveNet MediaPipe world landmarks` | `left_ear` | x | -0.14920 | 0.28328 | 0.02973 | 0.14569 | -0.02482 | -0.12638 | -0.10023 | 0.20591 | 0.23781 |
| `DriveNet MediaPipe world landmarks` | `left_ear` | y | -0.54570 | -0.42289 | -0.50002 | 0.03686 | -0.51986 | -0.53586 | -0.53048 | -0.45764 | -0.44347 |
| `DriveNet MediaPipe world landmarks` | `left_ear` | z | -0.24801 | -0.06588 | -0.17520 | 0.03610 | -0.18788 | -0.21223 | -0.19899 | -0.16383 | -0.10977 |
| `DriveNet MediaPipe world landmarks` | `right_ear` | x | -0.15345 | 0.19017 | 0.00888 | 0.11513 | -0.02748 | -0.12010 | -0.09626 | 0.15649 | 0.17111 |
| `DriveNet MediaPipe world landmarks` | `right_ear` | y | -0.52635 | -0.39466 | -0.47345 | 0.04045 | -0.49724 | -0.51051 | -0.50654 | -0.42290 | -0.41336 |
| `DriveNet MediaPipe world landmarks` | `right_ear` | z | -0.11983 | 0.06986 | -0.02714 | 0.03166 | -0.03438 | -0.06038 | -0.04898 | -0.01408 | 0.02446 |
| `DriveNet MediaPipe world landmarks` | `mouth_left` | x | -0.24511 | 0.21748 | -0.04306 | 0.15396 | -0.09870 | -0.20535 | -0.18433 | 0.14703 | 0.17553 |
| `DriveNet MediaPipe world landmarks` | `mouth_left` | y | -0.53561 | -0.38448 | -0.48017 | 0.04324 | -0.50331 | -0.52159 | -0.51437 | -0.43145 | -0.41306 |
| `DriveNet MediaPipe world landmarks` | `mouth_left` | z | -0.20752 | -0.02335 | -0.10641 | 0.02701 | -0.11284 | -0.13434 | -0.12558 | -0.08942 | -0.06814 |
| `DriveNet MediaPipe world landmarks` | `mouth_right` | x | -0.23752 | 0.17394 | -0.05123 | 0.13663 | -0.09954 | -0.19471 | -0.17696 | 0.11916 | 0.14443 |
| `DriveNet MediaPipe world landmarks` | `mouth_right` | y | -0.50468 | -0.35607 | -0.45064 | 0.04351 | -0.47477 | -0.49051 | -0.48563 | -0.40027 | -0.38338 |
| `DriveNet MediaPipe world landmarks` | `mouth_right` | z | -0.20178 | -0.01007 | -0.09063 | 0.02649 | -0.09624 | -0.11640 | -0.10896 | -0.07453 | -0.05230 |
| `DriveNet MediaPipe world landmarks` | `left_shoulder` | x | -0.08454 | 0.32065 | 0.11051 | 0.12957 | 0.08495 | -0.03993 | -0.00840 | 0.26549 | 0.29144 |
| `DriveNet MediaPipe world landmarks` | `left_shoulder` | y | -0.40608 | -0.24241 | -0.33488 | 0.05153 | -0.36167 | -0.38152 | -0.37504 | -0.27223 | -0.25599 |
| `DriveNet MediaPipe world landmarks` | `left_shoulder` | z | -0.23678 | -0.04288 | -0.15261 | 0.04047 | -0.16821 | -0.18814 | -0.17930 | -0.14282 | -0.07462 |
| `DriveNet MediaPipe world landmarks` | `right_shoulder` | x | -0.17797 | 0.24048 | 0.05650 | 0.11856 | 0.04589 | -0.08724 | -0.06754 | 0.19364 | 0.22242 |
| `DriveNet MediaPipe world landmarks` | `right_shoulder` | y | -0.47976 | -0.35868 | -0.43065 | 0.03897 | -0.45160 | -0.46723 | -0.45772 | -0.38377 | -0.36977 |
| `DriveNet MediaPipe world landmarks` | `right_shoulder` | z | 0.06165 | 0.22617 | 0.16058 | 0.02336 | 0.15652 | 0.13604 | 0.14520 | 0.17124 | 0.19807 |
| `DriveNet MediaPipe world landmarks` | `left_elbow` | x | -0.23212 | 0.16952 | 0.01490 | 0.10175 | 0.04490 | -0.13581 | -0.08219 | 0.09135 | 0.14248 |
| `DriveNet MediaPipe world landmarks` | `left_elbow` | y | -0.31134 | -0.08564 | -0.15751 | 0.03848 | -0.15124 | -0.21055 | -0.16518 | -0.13535 | -0.11815 |
| `DriveNet MediaPipe world landmarks` | `left_elbow` | z | -0.29916 | -0.11374 | -0.21033 | 0.04231 | -0.20539 | -0.26811 | -0.24565 | -0.18530 | -0.14711 |
| `DriveNet MediaPipe world landmarks` | `right_elbow` | x | -0.24735 | 0.01451 | -0.07339 | 0.07082 | -0.04166 | -0.17776 | -0.15582 | -0.01560 | -0.00246 |
| `DriveNet MediaPipe world landmarks` | `right_elbow` | y | -0.32511 | -0.15147 | -0.23013 | 0.02613 | -0.23850 | -0.25811 | -0.24938 | -0.20845 | -0.18945 |
| `DriveNet MediaPipe world landmarks` | `right_elbow` | z | 0.08618 | 0.27464 | 0.19622 | 0.03641 | 0.19496 | 0.14844 | 0.17072 | 0.22776 | 0.24680 |
| `DriveNet MediaPipe world landmarks` | `left_wrist` | x | -0.46265 | -0.08066 | -0.22143 | 0.08931 | -0.19438 | -0.34935 | -0.31248 | -0.15197 | -0.10408 |
| `DriveNet MediaPipe world landmarks` | `left_wrist` | y | -0.40129 | -0.07801 | -0.18155 | 0.06471 | -0.17271 | -0.26366 | -0.19295 | -0.15083 | -0.09490 |
| `DriveNet MediaPipe world landmarks` | `left_wrist` | z | -0.24509 | 0.00043 | -0.09996 | 0.05504 | -0.09164 | -0.17564 | -0.13979 | -0.05191 | -0.03540 |
| `DriveNet MediaPipe world landmarks` | `right_wrist` | x | -0.45911 | -0.19432 | -0.29122 | 0.07776 | -0.25575 | -0.40661 | -0.38169 | -0.22774 | -0.21489 |
| `DriveNet MediaPipe world landmarks` | `right_wrist` | y | -0.40472 | -0.05885 | -0.30590 | 0.05805 | -0.32593 | -0.36223 | -0.34646 | -0.27806 | -0.20701 |
| `DriveNet MediaPipe world landmarks` | `right_wrist` | z | -0.05437 | 0.30501 | 0.15223 | 0.07532 | 0.13308 | 0.06703 | 0.10011 | 0.22237 | 0.25881 |
| `DriveNet MediaPipe world landmarks` | `left_pinky` | x | -0.53222 | -0.13110 | -0.27973 | 0.09209 | -0.25518 | -0.40343 | -0.37396 | -0.20900 | -0.15695 |
| `DriveNet MediaPipe world landmarks` | `left_pinky` | y | -0.43893 | -0.07830 | -0.19911 | 0.07394 | -0.19047 | -0.29353 | -0.21403 | -0.16770 | -0.09422 |
| `DriveNet MediaPipe world landmarks` | `left_pinky` | z | -0.23490 | 0.02694 | -0.08277 | 0.05761 | -0.08035 | -0.15934 | -0.12429 | -0.03113 | -0.01270 |
| `DriveNet MediaPipe world landmarks` | `right_pinky` | x | -0.48063 | -0.20571 | -0.30859 | 0.07842 | -0.27419 | -0.42350 | -0.40139 | -0.24444 | -0.23089 |
| `DriveNet MediaPipe world landmarks` | `right_pinky` | y | -0.42720 | -0.04466 | -0.32361 | 0.06713 | -0.34255 | -0.39530 | -0.36857 | -0.29012 | -0.20861 |
| `DriveNet MediaPipe world landmarks` | `right_pinky` | z | -0.09234 | 0.29924 | 0.12971 | 0.08042 | 0.10819 | 0.04158 | 0.07614 | 0.20296 | 0.24360 |
| `DriveNet MediaPipe world landmarks` | `left_index` | x | -0.53104 | -0.16051 | -0.30349 | 0.08406 | -0.27932 | -0.41433 | -0.39062 | -0.23433 | -0.19489 |
| `DriveNet MediaPipe world landmarks` | `left_index` | y | -0.46143 | -0.11198 | -0.23258 | 0.07065 | -0.22445 | -0.32624 | -0.24432 | -0.20022 | -0.13655 |
| `DriveNet MediaPipe world landmarks` | `left_index` | z | -0.22458 | 0.03038 | -0.08131 | 0.05753 | -0.08145 | -0.15692 | -0.12177 | -0.02760 | -0.00897 |
| `DriveNet MediaPipe world landmarks` | `right_index` | x | -0.47748 | -0.20390 | -0.30531 | 0.08035 | -0.26733 | -0.42324 | -0.40268 | -0.24155 | -0.22505 |
| `DriveNet MediaPipe world landmarks` | `right_index` | y | -0.48569 | -0.07899 | -0.37437 | 0.06911 | -0.38559 | -0.45433 | -0.42505 | -0.33895 | -0.26010 |
| `DriveNet MediaPipe world landmarks` | `right_index` | z | -0.09379 | 0.29238 | 0.12224 | 0.08288 | 0.09484 | 0.03153 | 0.06348 | 0.20062 | 0.24105 |
| `DriveNet MediaPipe world landmarks` | `left_thumb` | x | -0.45918 | -0.10309 | -0.23545 | 0.08169 | -0.20901 | -0.35307 | -0.31990 | -0.16932 | -0.13165 |
| `DriveNet MediaPipe world landmarks` | `left_thumb` | y | -0.40564 | -0.07626 | -0.18591 | 0.06635 | -0.17742 | -0.27059 | -0.19756 | -0.15618 | -0.09511 |
| `DriveNet MediaPipe world landmarks` | `left_thumb` | z | -0.23892 | 0.00657 | -0.09533 | 0.05502 | -0.08859 | -0.17065 | -0.13416 | -0.04672 | -0.02974 |
| `DriveNet MediaPipe world landmarks` | `right_thumb` | x | -0.47794 | -0.20733 | -0.30443 | 0.08117 | -0.26616 | -0.42453 | -0.40129 | -0.24020 | -0.22382 |
| `DriveNet MediaPipe world landmarks` | `right_thumb` | y | -0.42370 | -0.06162 | -0.32891 | 0.05993 | -0.34897 | -0.38923 | -0.36918 | -0.30362 | -0.22590 |
| `DriveNet MediaPipe world landmarks` | `right_thumb` | z | -0.06620 | 0.29621 | 0.14020 | 0.07762 | 0.11723 | 0.05480 | 0.08575 | 0.21361 | 0.25056 |
| `DriveNet MediaPipe world landmarks` | `left_hip` | x | -0.02383 | 0.02984 | 0.01159 | 0.00925 | 0.01148 | 0.00288 | 0.00703 | 0.01654 | 0.02374 |
| `DriveNet MediaPipe world landmarks` | `left_hip` | y | 0.01204 | 0.04293 | 0.02599 | 0.00854 | 0.02469 | 0.01566 | 0.01843 | 0.03511 | 0.03785 |
| `DriveNet MediaPipe world landmarks` | `left_hip` | z | -0.07435 | -0.05210 | -0.06489 | 0.00626 | -0.06668 | -0.07187 | -0.07053 | -0.05816 | -0.05616 |
| `DriveNet MediaPipe world landmarks` | `right_hip` | x | -0.02465 | 0.02579 | -0.00874 | 0.00849 | -0.00857 | -0.01916 | -0.01403 | -0.00507 | -0.00034 |
| `DriveNet MediaPipe world landmarks` | `right_hip` | y | -0.04607 | -0.01589 | -0.03094 | 0.00781 | -0.03093 | -0.04135 | -0.03855 | -0.02406 | -0.02080 |
| `DriveNet MediaPipe world landmarks` | `right_hip` | z | 0.05250 | 0.07448 | 0.06609 | 0.00545 | 0.06677 | 0.05891 | 0.06064 | 0.07147 | 0.07299 |
| `DriveNet MediaPipe world landmarks` | `left_knee` | x | -0.40666 | -0.33317 | -0.36779 | 0.01930 | -0.36572 | -0.39628 | -0.38315 | -0.35098 | -0.34435 |
| `DriveNet MediaPipe world landmarks` | `left_knee` | y | -0.11581 | 0.11307 | 0.02933 | 0.05690 | 0.05109 | -0.05756 | -0.01418 | 0.07340 | 0.08627 |
| `DriveNet MediaPipe world landmarks` | `left_knee` | z | -0.17732 | -0.04816 | -0.12810 | 0.02355 | -0.13031 | -0.16073 | -0.14317 | -0.11048 | -0.10153 |
| `DriveNet MediaPipe world landmarks` | `right_knee` | x | -0.39016 | -0.27525 | -0.31736 | 0.01913 | -0.31258 | -0.34704 | -0.33060 | -0.30335 | -0.29697 |
| `DriveNet MediaPipe world landmarks` | `right_knee` | y | -0.17317 | 0.00093 | -0.11825 | 0.01830 | -0.11881 | -0.14031 | -0.13164 | -0.10674 | -0.09687 |
| `DriveNet MediaPipe world landmarks` | `right_knee` | z | -0.01080 | 0.18541 | 0.06953 | 0.04403 | 0.06579 | 0.01455 | 0.03171 | 0.10478 | 0.13133 |
| `DriveNet MediaPipe world landmarks` | `left_ankle` | x | -0.61793 | -0.24260 | -0.52252 | 0.04827 | -0.52751 | -0.57576 | -0.54639 | -0.50203 | -0.47164 |
| `DriveNet MediaPipe world landmarks` | `left_ankle` | y | 0.07940 | 0.44462 | 0.31940 | 0.05804 | 0.33197 | 0.24169 | 0.28029 | 0.35785 | 0.38361 |
| `DriveNet MediaPipe world landmarks` | `left_ankle` | z | -0.08948 | 0.22022 | 0.02779 | 0.05428 | 0.02279 | -0.04038 | -0.01059 | 0.06303 | 0.09811 |
| `DriveNet MediaPipe world landmarks` | `right_ankle` | x | -0.63122 | -0.24896 | -0.52890 | 0.03941 | -0.52733 | -0.57910 | -0.54913 | -0.50767 | -0.48031 |
| `DriveNet MediaPipe world landmarks` | `right_ankle` | y | 0.07239 | 0.29195 | 0.20573 | 0.03437 | 0.20714 | 0.16763 | 0.18819 | 0.21983 | 0.24917 |
| `DriveNet MediaPipe world landmarks` | `right_ankle` | z | 0.07299 | 0.30480 | 0.15654 | 0.04371 | 0.15446 | 0.09898 | 0.12484 | 0.18334 | 0.20986 |
| `DriveNet MediaPipe world landmarks` | `left_heel` | x | -0.66905 | -0.24999 | -0.55673 | 0.05368 | -0.55873 | -0.61776 | -0.58106 | -0.53555 | -0.50124 |
| `DriveNet MediaPipe world landmarks` | `left_heel` | y | 0.07748 | 0.46560 | 0.34141 | 0.05848 | 0.35572 | 0.26650 | 0.30333 | 0.37932 | 0.40437 |
| `DriveNet MediaPipe world landmarks` | `left_heel` | z | -0.07388 | 0.26134 | 0.05399 | 0.05862 | 0.04836 | -0.01947 | 0.01270 | 0.09285 | 0.13052 |
| `DriveNet MediaPipe world landmarks` | `right_heel` | x | -0.66630 | -0.24262 | -0.54823 | 0.04390 | -0.54606 | -0.60444 | -0.56861 | -0.52618 | -0.49436 |
| `DriveNet MediaPipe world landmarks` | `right_heel` | y | 0.08124 | 0.32193 | 0.23052 | 0.03773 | 0.23224 | 0.18725 | 0.21120 | 0.24727 | 0.27503 |
| `DriveNet MediaPipe world landmarks` | `right_heel` | z | 0.03755 | 0.27735 | 0.12572 | 0.04523 | 0.12429 | 0.06730 | 0.09152 | 0.15309 | 0.18246 |
| `DriveNet MediaPipe world landmarks` | `left_foot_index` | x | -0.82777 | -0.34918 | -0.68341 | 0.06080 | -0.67862 | -0.76148 | -0.70872 | -0.65663 | -0.63306 |
| `DriveNet MediaPipe world landmarks` | `left_foot_index` | y | 0.05684 | 0.46287 | 0.32972 | 0.06441 | 0.34176 | 0.24796 | 0.28627 | 0.37114 | 0.40381 |
| `DriveNet MediaPipe world landmarks` | `left_foot_index` | z | -0.08457 | 0.38324 | 0.08889 | 0.07926 | 0.07690 | -0.00565 | 0.03666 | 0.14268 | 0.18770 |
| `DriveNet MediaPipe world landmarks` | `right_foot_index` | x | -0.75476 | -0.32582 | -0.62605 | 0.04766 | -0.62347 | -0.69223 | -0.64675 | -0.60314 | -0.56760 |
| `DriveNet MediaPipe world landmarks` | `right_foot_index` | y | -0.01684 | 0.32727 | 0.21647 | 0.06159 | 0.21992 | 0.13917 | 0.17526 | 0.26640 | 0.29569 |
| `DriveNet MediaPipe world landmarks` | `right_foot_index` | z | 0.03370 | 0.31630 | 0.14491 | 0.06021 | 0.14039 | 0.06625 | 0.09895 | 0.18660 | 0.22742 |

## Interpretation

The paper explicitly describes MediaPipe **World Landmarks** in meters, with hip-midpoint origin, while the normalized image landmark output has x/y in image-normalized coordinates and a different z convention. The local CSV has 99 coordinate columns; therefore both the paper-level representation and the measured file-level serialization are recorded separately.
