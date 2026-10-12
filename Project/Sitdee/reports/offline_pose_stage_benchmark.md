# Offline stage-segmented pose benchmark

MEASURED — local recorded video only; no camera/RTSP, no posture accuracy.
Video: `datasets\raw\rgb_native\drivenet_v6\source_videos\SS_1.mp4`; frames/run: 90; runs: 3
Environment: `3.12.10`; GPU: NVIDIA GeForce GTX 1660 SUPER

## YOLO + ByteTrack (measured)
Steady FPS: {'mean': 45.60353309087998, 'median': 47.51484178881583, 'p95': 49.40814146973141}; stage ms/frame: {'mean': 22.12003814789501, 'median': 20.133949990849942, 'p95': 33.37311999930534}; peak allocated VRAM MiB: 68.1123046875

## MediaPipe CPU (measured)
GPU delegate: GPU_DELEGATE_UNAVAILABLE_OR_FAILED
All N>1 results duplicate one valid crop as SYNTHETIC_COMPUTE_LOAD, not multi-person accuracy.
N=1: event ms {'mean': 92.76703749492299, 'median': 91.29705000668764, 'p95': 106.81790999660734}; ms/crop {'mean': 92.73826250137063, 'median': 91.27980000630487, 'p95': 106.76774501189357}; pose successes 8/8
N=2: event ms {'mean': 145.069300004252, 'median': 143.826049999916, 'p95': 151.1065850078012}; ms/crop {'mean': 72.52115000119375, 'median': 71.56324999232311, 'p95': 77.85522500489606}; pose successes 16/16
N=4: event ms {'mean': 287.6162749998912, 'median': 286.7073499946855, 'p95': 292.81060501089087}; ms/crop {'mean': 71.89035937699373, 'median': 71.65559999702964, 'p95': 74.74382499349304}; pose successes 32/32
N=8: event ms {'mean': 599.5572125066246, 'median': 593.5908000101335, 'p95': 635.3696100108209}; ms/crop {'mean': 74.93225937423631, 'median': 72.66259999596514, 'p95': 84.29631000326481}; pose successes 64/64

## Combined offline replay (measured)
Sampling is scheduled on video timestamps; throughput is measured wall-clock playback speed, not network latency.
The default 90-frame window is about 3 seconds of video time: low-rate cases have only 2-3 sample events. Per-frame p95 can therefore hide rare long pose events; inspect pose-event p95 and total-frame maximum as well.
N1_rate0.5: FPS 47.45610244587656; YOLO ms {'mean': 19.451402222491904, 'median': 18.774750002194196, 'p95': 23.87902999471407}; Pose ms/event {'mean': 71.87669999257196, 'median': 71.87669999257196, 'p95': 72.45323999522952}; total ms/frame {'mean': 21.069883333661387, 'median': 18.79699999699369, 'p95': 24.7741950006457}; total max 95.2 ms; attempts 2
N1_rate1: FPS 43.25932480030572; YOLO ms {'mean': 20.419921111250815, 'median': 19.117550007649697, 'p95': 26.23135499889031}; Pose ms/event {'mean': 80.08120000401202, 'median': 80.77969998703338, 'p95': 86.4073100121459}; total ms/frame {'mean': 23.114113334385266, 'median': 19.1651500063017, 'p95': 29.863279989513103}; total max 113.4 ms; attempts 3
N1_rate2: FPS 39.94128630946202; YOLO ms {'mean': 20.042171112598023, 'median': 19.212850005715154, 'p95': 25.750044999585953}; Pose ms/event {'mean': 74.37830000708345, 'median': 72.7455499873031, 'p95': 81.4630250170012}; total ms/frame {'mean': 25.034432221501547, 'median': 19.252749989391305, 'p95': 89.59795998089248}; total max 104.3 ms; attempts 6
N2_rate0.5: FPS 43.435431854935224; YOLO ms {'mean': 19.60515999784952, 'median': 18.99920000869315, 'p95': 25.049434993707106}; Pose ms/event {'mean': 152.73300000990275, 'median': 152.73300000990275, 'p95': 159.4700400091824}; total ms/frame {'mean': 23.02043333248649, 'median': 19.04104999266565, 'p95': 25.749449993600138}; total max 178.7 ms; attempts 4
N2_rate1: FPS 38.9095991235193; YOLO ms {'mean': 20.514326667438986, 'median': 19.453300003078766, 'p95': 26.030205011193175}; Pose ms/event {'mean': 154.76550000797337, 'median': 144.19640001142398, 'p95': 172.8259400028037}; total ms/frame {'mean': 25.698254444351832, 'median': 19.53069999581203, 'p95': 29.18346500810002}; total max 202.7 ms; attempts 6
N2_rate2: FPS 34.28689244887498; YOLO ms {'mean': 19.50052444557918, 'median': 18.895199988037348, 'p95': 23.12019999953918}; Pose ms/event {'mean': 144.41626666424176, 'median': 144.00635000492912, 'p95': 147.06452498649014}; total ms/frame {'mean': 29.16349888805093, 'median': 18.943999995826744, 'p95': 162.15504501451505}; total max 169.3 ms; attempts 12
N4_rate0.5: FPS 37.84449051286786; YOLO ms {'mean': 19.935976667006294, 'median': 19.248050011810847, 'p95': 24.905755007057447}; Pose ms/event {'mean': 290.87964999780525, 'median': 290.87964999780525, 'p95': 292.51013498433167}; total ms/frame {'mean': 26.42151111342375, 'median': 19.313900003908202, 'p95': 26.522950002981815}; total max 314.6 ms; attempts 8
N4_rate1: FPS 33.94462319427121; YOLO ms {'mean': 19.538220000039374, 'median': 18.764699998428114, 'p95': 24.53628000512253}; Pose ms/event {'mean': 296.85043334029615, 'median': 293.5005000035744, 'p95': 304.670940010692}; total ms/frame {'mean': 29.45739222276542, 'median': 18.800750010996126, 'p95': 25.91927499306621}; total max 325.4 ms; attempts 12
N4_rate2: FPS 25.180626227641902; YOLO ms {'mean': 20.415734446012518, 'median': 19.205750009859912, 'p95': 26.448825004626997}; Pose ms/event {'mean': 288.9327500015497, 'median': 288.0169499985641, 'p95': 294.6041000104742}; total ms/frame {'mean': 39.71082555718668, 'median': 19.39515001140535, 'p95': 307.0408350031357}; total max 315.6 ms; attempts 24
N8_rate0.5: FPS 30.47586540052071; YOLO ms {'mean': 19.799228891174103, 'median': 19.06889998645056, 'p95': 23.716729989973828}; Pose ms/event {'mean': 584.5433500071522, 'median': 584.5433500071522, 'p95': 591.8291650072206}; total ms/frame {'mean': 32.81058222023098, 'median': 19.115950010018423, 'p95': 25.079359996016134}; total max 612.5 ms; attempts 16
N8_rate1: FPS 25.726883066210952; YOLO ms {'mean': 19.41463222158038, 'median': 18.78089999081567, 'p95': 22.848709992831573}; Pose ms/event {'mean': 582.8925333238052, 'median': 582.4798999819905, 'p95': 588.9500000048429}; total ms/frame {'mean': 38.86754777502372, 'median': 18.7939499883214, 'p95': 24.638660004711703}; total max 608.7 ms; attempts 24
N8_rate2: FPS 16.696128221683217; YOLO ms {'mean': 20.202456664992496, 'median': 19.293750010547228, 'p95': 24.670799997693393}; Pose ms/event {'mean': 594.8118666710798, 'median': 590.6222000048729, 'p95': 621.9328000006499}; total ms/frame {'mean': 59.8916644443913, 'median': 19.442000004346482, 'p95': 600.2006300084759}; total max 649.7 ms; attempts 48

Initial measured-budget recommendation: at most 2.0 pose crops/second globally (experimental, not production default).
Reason: Experimental global start: measured N<=4 cases retaining >=70% of tracking-only median FPS and <=250 ms p95 total frame time; capped at 2 crops/s. Revalidate on real CCTV before production.
No V4/posture classification or labels were used.
