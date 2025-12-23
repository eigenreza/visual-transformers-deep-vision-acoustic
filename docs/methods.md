# Methods

The methods follow the order of the pipeline: the annotation dataset, the detector with its calibration and operating points, the inference on the camera transects, and the camera depth and time series. The last section describes in brief the infrastructure prepared for the integration with the echosounder record. `docs/data.md` documents the data, and `docs/results.md` holds the numbers that each method produced.

## 1. Annotation dataset

### 1.1 Pairing

Images and Pascal VOC files are paired by their path relative to the respective tree root with the extension removed. One tow stores its XML files one directory level above its images; those files are paired by tow and file stem and marked as such. XML files that reference frames of another tow (relabelling sets) are left unpaired. Image size is taken from the XML file and, where the size fields are missing, from the image header. The nested duplicate copies of both trees are excluded.

### 1.2 Taxonomy

Raw labels are mapped to classes by exact string match through `configs/taxonomy.yaml`; a raw label absent from the mapping stops the build, so a new label in the collection cannot pass unnoticed. A class is kept when the collection holds at least about two hundred boxes of it, the smallest count for which per-class precision and recall carry useful intervals. Raw labels that differ only in capitalisation or spelling denote the same taxon and are merged; `periphylla` and `jellyfish` join the group class `gelatinous`. This gives thirteen classes: maurolicus, krill, benthosema, redfish, bluewhiting, gelatinous, mackerel, herring, gonatus, sergestes, capelin, pasiphaea and barracudina.

A fourteenth class, `unidentified`, collects the labels annotators used for objects they could not identify (`UID`, `Uid_fish`, `undefined`, `unknown`, `mesopelagic`, `mesopelagic_fish`, `large_fish`) together with taxa of fewer than sixty boxes (gadoids, cod, copepods, mysids, other crustaceans). Deleting these boxes would teach the detector to treat real animals as background. The class is trained like any other, and its per-frame count is kept in the inference tables as a measure of how much the taxonomy misses.

### 1.3 Negatives

Annotation files without boxes are images the annotator examined and found empty, provided the surrounding material was annotated. A tow or folder in which no frame has a box is treated as unannotated and excluded. In this collection every zero-box file sits in a folder that also holds annotated frames, so all 429 are kept as negatives.

### 1.4 Splits

The unit of assignment is the tow, so that consecutive and near-identical frames never fall on both sides of a split boundary. All material of 2024 forms an out-of-distribution test set and enters no other split. Stations ST12, ST13 and ST14 of 2018 were recorded during the transects that the detector is later applied to; they are forced into the training split. A check by frame timestamp against the transect time windows confirms that no frame from those windows sits in validation or test. Because the detector is applied to 2018 imagery, its performance on 2018 must be measurable: validation and test are each required to contain at least five further 2018 stations.

The remaining tows are assigned by a seeded search. Each start fills the splits greedily with shuffled tows to image quotas of 70, 15 and 15 percent. It then runs 6,000 proposals that move one tow or swap two tows between splits; improvements are always accepted, deteriorations with a probability that decays geometrically. The loss sums six terms:

- the excess of each split's image fraction over its target beyond two percent, weighted by ten;
- five for every 2018 station missing from the required minimum in validation or test;
- for every class, the absolute deviation of its box fraction in every split from the target, with a quarter weight for classes carried by fewer than three tows;
- twice the shortfall of every class below half of its boxes in training;
- twice the shortfall of every class carried by at least three tows below the larger of ten boxes and five percent in validation and in test, with a class carried by two tows required in one of them;
- the excess of each year's image fraction beyond three percent, for years with at least 200 images.

Forty starts are run and the best assignment is kept. Two seeds produce the same partition up to the exchange of validation and test.

Classes that live in one or two tows cannot be balanced: gonatus has 99 percent of its boxes in one tow, capelin has two tows of similar size, and redfish has one tow with 72 percent of its boxes. The search keeps the dominant tow of each in training and places the rest where the other classes allow. The resulting assignment is frozen in `configs/splits/annotation_splits.csv` and verified by automated checks before every export.

### 1.5 Export

Each split is written as one COCO file. Image paths are relative to the image root of the collection, category identifiers are the one-based positions in the class list, boxes are clipped to the image, and boxes with a side under two pixels are dropped. Images without boxes are included as negatives.

## 2. Detector

Earlier work on Deep Vision imagery trained convolutional networks to identify fish species (Allken et al., 2019), to identify and count pelagic and mesopelagic fishes (Allken et al., 2021) and to segment fish for size measurement (Garcia et al., 2020). The detector here is a detection transformer trained on the annotation collection of Section 1.

### 2.1 Candidates and licences

Three detector families are compared on identical splits, input sizes and evaluation code. RT-DETRv2 (Zhao et al., 2024; Lv et al., 2024) and D-FINE (Peng et al., 2024) are real-time detection transformers trained through their original repositories (both Apache-2.0), starting from checkpoints pretrained on Objects365 (Shao et al., 2019) and COCO (Lin et al., 2014). The convolutional reference is RetinaNet (Lin et al., 2017) with a ResNet-50 feature pyramid backbone from torchvision (BSD-3-Clause), a widely used one-stage convolutional detector, started from COCO weights with a new classification head. The frameworks index class outputs by category identifier, so the models are built with fifteen outputs of which identifier zero is never used.

### 2.2 Training recipe

All models train with mixed precision, AdamW, a linear warm-up of 500 iterations, gradient clipping and a fixed seed. The learning rate follows the framework's schedule: constant with an exponential moving average of the weights for the transformers, cosine decay for RetinaNet. Augmentation is the same for every model: photometric distortion with probability one half, an IoU-constrained random crop with probability one half, horizontal flips and, for the transformers, multi-scale training between 75 and 125 percent of the input size. Zoom-out is not used because it shrinks the smallest classes further. The strong augmentations are switched off for the last six epochs. Training runs for 40 epochs; the checkpoint with the best validation mAP is kept. Input sizes of 640 and 1,024 pixels are compared on one model first; 800 is added if the two differ markedly.

### 2.3 Evaluation

Five classes occur in the 2018 annotations: krill, benthosema, blue whiting, mackerel and herring. They are called the primary classes (`label_classes` in the configuration). The input size is decided on krill, the most numerous primary class in the transects and the one most sensitive to resolution. The 2018 test stations hold no krill, so the decision is read on other material. The larger input is adopted when it beats the smaller one on krill AP50 on the 2020 to 2023 test slice or on the 2024 hold-out, with a paired bootstrap interval that excludes zero. Krill recall at the operating threshold is reported alongside.

Krill and blue whiting have no boxes in the 2018 test stations, so neither can be evaluated on 2018 test imagery. Their detector quality is read from the 2022 to 2023 test material and from the frame-level review of transect detections (Section 3).

Every model writes its detections for a split as a table of boxes in original image pixels, with a score threshold of 0.001 and at most 300 detections per image. All metrics are computed from that table by the project's own code. Detections are matched greedily by score to ground truth of the same class at an intersection over union of 0.5. Reported quantities are COCO mAP50:95 and mAP50 through faster-coco-eval, per-class AP50 from all-point interpolation, and precision, recall and F1 against the confidence threshold. The operating threshold of a class is the one that maximises its F1 on validation, and a confusion matrix with a background row and column is computed at those thresholds. The AP50 figures are repeated for slices of images by year (2017, 2018, 2020 to 2023 pooled and individually, 2024). Intervals are percentile bootstraps over images with 200 resamples; the same resamples are used for all classes and models of a split so that differences between models can be read from paired draws. The 2018 slice is the figure of merit, because the detector is applied to the 2018 transects.

### 2.4 Calibration, thresholds and choice of detector

Two rules for the existence of an object are compared on validation. Under the per-class rule an object exists when the highest calibrated score of its query clears the operating threshold of that class. Under the pooled rule the five primary classes share one threshold, chosen on the validation split to maximise the F1 of "any primary-class object" (a primary-class object matched to a box of any of the five classes at IoU 0.5). The other classes keep their own thresholds. The class is assigned by the top hypothesis, and the cross-class suppression and the expected-count split are unchanged. The per-class thresholds remain the operating points of the unresolved hard counts. The rule with the higher macro F1 over the primary classes on the validation split is used for inference, and both rules are reported on the 2018 validation, 2018 test and 2020 to 2023 test slices. For both transformers at 1,024 pixels the per-class rule has the higher validation macro F1 (RT-DETRv2-L 0.805 against 0.787 at a pooled threshold of 0.47; D-FINE-L 0.793 against 0.781 at 0.45), so inference uses the per-class rule. On validation the pooled rule buys a few points of herring and blue whiting recall at a larger loss of precision. Herring recall at the operating point on the 2018 test stations is 0.88 for RT-DETRv2-L and 0.48 for D-FINE-L.

A detection's confidence is calibrated to the probability that it is a true positive at the evaluation matching. Validation detections are labelled by matching, and three monotone maps of the raw score are fitted: a temperature on the logit (Guo et al., 2017), Platt scaling (Platt, 1999) and isotonic regression. Each map is also fitted on the 2018 validation stations alone, and negative log likelihood, expected calibration error and reliability tables are reported per slice before and after calibration. The map used for transect inference is chosen from these tables: fitted on the whole validation split when the 2018 slice agrees with it, and on the 2018 slice when it does not. Per-class operating thresholds are then re-derived on calibrated scores.

The detector for transect inference is chosen between RT-DETRv2-L and D-FINE-L at 1,024 pixels, the two transformers trained under the recipe of Section 2.2; RetinaNet is the convolutional reference. Validation alone does not separate the two transformers. The macro mAP50 over the five primary classes on the whole validation split differs by +0.008 in favour of RT-DETRv2-L, with a paired interval of [-0.019, +0.039]. That is smaller than the gap between two D-FINE-L seeds (0.808 and 0.823, against 0.816 for RT-DETRv2-L). The macro F1 of the primary classes at the operating points is 0.805, against 0.793 and 0.806 for the two D-FINE-L seeds. On the 2018 validation stations (28 images, five stations) both models count 12 of the 12 herring boxes at precision and recall 0.83 and all 16 blue whiting boxes, at precision 0.76 against 0.70. The AP50 differences for the two species (-0.04 and -0.05) have intervals that span zero. The only validation intervals that exclude zero point in opposite directions: krill recall at the operating threshold is higher for RT-DETRv2-L by 0.07 [0.05, 0.11] and benthosema AP50 is lower by 0.03 [0.01, 0.05]. The choice therefore used the test split. It rested on two quantities. On the 2018 test stations, herring recall at the operating point is 0.88 against 0.48, and 1 against 17 of the 48 boxes are taken for blue whiting. On the whole test split, the primary-class macro mAP50 differs by +0.030 [+0.010, +0.052]. The test metrics of the selected detector are selection-biased as a result. The 2024 cruise did not enter this choice and is the only hold-out whose figures are free of it. The input-size rule of Section 2.3 read krill AP50 on the 2024 cruise once; that is the only other use of that hold-out.

## 3. Transect inference

The chosen detector runs on every frame of the left camera of the four 2018 transects at the input size it was trained with. Detections with raw score at or above 0.05 are stored with their class, raw and calibrated confidence, box in frame pixels and frame timestamp, so that any later threshold can be applied without rerunning inference. A second table gives, per frame, the number of detections per class above a grid of raw thresholds (0.05, 0.1, 0.2, 0.3, 0.5) and above the per-class operating thresholds on calibrated confidence, including the unidentified class as a diagnostic. Folders are processed one at a time and skipped when already done, so an interrupted run resumes. Quality control records the no-detection rate per folder at the floor and at the operating thresholds, the confidence quantiles per class and the throughput.

Near the surface in daylight the camera sees its own housing and the warp, and the detector labels them as organisms with a box that is the same in every frame. Boxes whose position and size, quantised to 16 pixel cells, recur in at least 20 percent of the frames of a folder are taken as templates of fixed structure, one template per recurring cell with the median box. Every detection of any class in the transect that overlaps a template at an intersection over union of 0.5 or more is marked static and excluded from all counts and from the review sheets. The number of static detections above the operating thresholds is kept per frame, and a camera interval in which more than half of the frames show one is flagged as spent at the surface. Confidence-weighted counts are the sums of the calibrated scores of the resolved objects; the same sum over every stored first hypothesis above the floor is kept as a diagnostic column.

A detection transformer scores every class for every query, and the stored detections are the highest-scoring (query, class) pairs above the floor, so one object can appear with two classes and the same box. Rows that share a box within a frame are given the same query identifier and a rank by score. Per-frame counts are formed three ways. Unresolved: every row at or above its class operating threshold counts. Resolved: an object exists when the highest calibrated score of its query clears the threshold of that class. Overlapping objects of different classes (intersection over union above 0.7) keep only the highest score, and each surviving object counts once in its highest class. Expected: each surviving object distributes one unit of count over its two highest classes in proportion to their calibrated scores. The expected counts therefore sum to the resolved object count and differ from it only in how objects are shared between confusable classes. The sum of calibrated scores of all stored hypotheses of rank one and two above the floor is kept as a diagnostic column, not as a count, because it adds fractions of objects rather than redistributing them. Before inference the resolution is applied to the annotated 2018 validation and test frames, and the unresolved, resolved and expected counts per class are compared with the boxes, together with the pairs of classes that suppress each other.

### 3.1 Frame-level review

Krill is the class whose detection quality is least constrained by annotated 2018 imagery. For each transect, a fixed random sample of 50 frames with krill is drawn in each of three bands of calibrated confidence (0.05 to 0.3, 0.3 to 0.6, 0.6 and above). A frame belongs to the band of its highest-confidence detection of the class. A fourth set of 50 frames holds maurolicus detections at or above 0.3 calibrated confidence. Krill missed at the operating threshold are mostly predicted as maurolicus, a class without 2018 annotations, and a reader should be able to judge whether those are krill. The frames are saved as contact sheets with the boxes of the class in red and other classes in grey, the sample is listed in a table, and the seed fixes the sample so that reviewers see the same frames.

## 4. Camera depth and time series

The Deep Vision metadata reports a depth per frame, updated about every 0.65 seconds in steps of 0.8 metres. Exact zeros mark dropouts (3 to 8 percent of frames). Consecutive readings differ by 5 metres in the median and by 20 metres at the 90th percentile, far more than a towed body moves in two thirds of a second. The readings therefore scatter around the true trajectory with an interquartile range of 5 to 10 metres and must be smoothed before use.

Cleaning proceeds in three steps. A centred rolling median over 30 seconds of the non-zero readings gives the physical trend (a 60 second window leaves a systematic residual of up to 1.5 m on the fast descent and ascent ramps; 30 seconds removes it). The 95th percentile of its speed over ten-second steps serves as the vertical speed bound. A reading is rejected when it departs from the trend by more than three robust scales of the residuals plus the distance covered at the speed bound within half a trend window. The speed bound, the tolerance and the rejection counts are reported per transect. The cleaned depth is a centred rolling median over 30 seconds of the consistent readings. It is interpolated in time across gaps shorter than ten seconds and left empty across longer ones, and its own speed stays below 0.45 m/s at the 95th percentile. Every frame keeps its raw reading, the trend, the cleaned depth, the residual, a reading flag (consistent, rejected, dropout) and the source of the cleaned value (median, interpolated, none).

Frames and detections are then aggregated to ten-second intervals. An interval holds the number of frames, the number with a cleaned depth, and the mean, median, minimum and maximum of the cleaned depth. It also holds the median and mean of the raw non-zero readings, the per-class counts above each threshold and above the operating thresholds, and confidence-weighted counts (the sum of calibrated confidences per class). Intervals without frames are kept as empty rows so that the series is regular in time.

The cleaner is examined per transect in four ways. The distribution of raw minus cleaned depth is tabulated, with the share of frames that differ by more than 2 m and 10 m. Isolated sensor updates are told from persistent offsets by the run lengths of readings more than 5 m from the trend, the correlation of consecutive update steps, and the single-update spikes that return to the preceding value. A lean to one side is read from the residual quantiles by depth and from the rejected readings deeper against shallower than the trend. A twenty-minute excerpt is drawn with raw and cleaned depth overlaid. The rejection band is symmetric about a centred median, so the cleaner itself has no preferred direction; a one-sided rejection count reports a one-sided raw distribution. Both the raw and the cleaned series are kept in the output tables.

## 5. Infrastructure for acoustic integration

The package also holds the steps that connect the camera series to the echosounder record of the same cruise. They are prepared for the future work named in the README and are described here in brief. No results are reported for them. The classification of echosounder data by machine learning, and its use in survey estimates, is an active field (Ordoñez et al., 2022; Yassir et al., 2023; Handegard et al., 2025).

### 5.1 Acoustic preprocessing

The store holds the linear volume backscattering coefficient per channel, ping and range sample. Values are converted to volume backscattering strength in dB with a floor; depth is range plus the transducer draft; channels are addressed by their frequency. Every sample carries a mask with one bit each for missing values, the surface exclusion below the transducer, the seabed with a margin, a low signal-to-noise ratio, and pings dominated by impulsive noise. The seabed is detected on the 38 kHz channel. Background noise is estimated per block of pings from its quietest range block, with a standard formula for the absorption of sea water, and subtracted in the linear domain. Impulsive noise is flagged by a two-sided comparison of neighbouring pings. Two products are written as Zarr: the full-resolution record around each camera transect, and the whole cruise averaged to cells of one ping by 1 m.

### 5.2 Alignment of camera and echosounder

The camera trails the vessel, so the echosounder passes over a volume of water before the camera reaches it. The horizontal distance between vessel and camera is not recorded in these data. The alignment command therefore bounds the lag from the vessel speed and a straight warp of an assumed range of lengths. It brings both sensors to the ten-second grid of the camera series and reads the backscatter in a band around the camera depth at every lag of a grid. The backscatter is correlated by rank with the camera signals, with block-bootstrap intervals and a permutation null for the maximum of the search. The output is a table that maps every camera interval to a vessel time, with the lag carried as a distribution over its physical range and with flags for intervals in which the camera depth lies inside the surface exclusion. A trawl position and a time base synchronised with the echosounder would replace the assumed geometry.

### 5.3 Window construction

Echogram patches are cut at the mapped vessel time in cells of fixed duration and height, with their mask bits and the depth of every cell row, in several geometries around the camera depth. Each patch is paired with the detection counts of the corresponding camera interval, in the count forms of Section 3, with the uncertainty of the lag carried into the pairing. Splits are made by transect, leaving one transect out, and within transects by contiguous time blocks with a buffer on both sides of every boundary. Windows never straddle a boundary, and a test asserts these properties on the split files.

## References

Allken, V., Handegard, N. O., Rosen, S., Schreyeck, T., Mahiout, T., & Malde, K. (2019). Fish species identification using a convolutional neural network trained on synthetic data. ICES Journal of Marine Science, 76(1), 342-349. https://doi.org/10.1093/icesjms/fsy147

Allken, V., Rosen, S., Handegard, N. O., & Malde, K. (2021). A deep learning-based method to identify and count pelagic and mesopelagic fishes from trawl camera images. ICES Journal of Marine Science, 78(10), 3780-3792. https://doi.org/10.1093/icesjms/fsab227

Garcia, R., Prados, R., Quintana, J., Tempelaar, A., Gracias, N., Rosen, S., Vågstøl, H., & Løvall, K. (2020). Automatic segmentation of fish using deep learning with application to fish size measurement. ICES Journal of Marine Science, 77(4), 1354-1366. https://doi.org/10.1093/icesjms/fsz186

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. Proceedings of the 34th International Conference on Machine Learning (ICML). arXiv:1706.04599

Handegard, N. O., Holmin, A. J., Pala, A., Utseth, I., & Johnsen, E. (2025). Integrating and assessing machine learning acoustic target classification models for fish survey estimations. ICES Journal of Marine Science, 82(5), fsaf069. https://doi.org/10.1093/icesjms/fsaf069

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. 2017 IEEE International Conference on Computer Vision (ICCV), 2999-3007. https://doi.org/10.1109/ICCV.2017.324

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. Computer Vision, ECCV 2014, Lecture Notes in Computer Science, 740-755. https://doi.org/10.1007/978-3-319-10602-1_48

Lv, W., Zhao, Y., Chang, Q., Huang, K., Wang, G., & Liu, Y. (2024). RT-DETRv2: Improved baseline with bag-of-freebies for real-time detection transformer. arXiv:2407.17140

Ordoñez, A., Utseth, I., Brautaset, O., Korneliussen, R., & Handegard, N. O. (2022). Evaluation of echosounder data preparation strategies for modern machine learning models. Fisheries Research, 254, 106411. https://doi.org/10.1016/j.fishres.2022.106411

Peng, Y., Li, H., Wu, P., Zhang, Y., Sun, X., & Wu, F. (2024). D-FINE: Redefine regression task in DETRs as fine-grained distribution refinement. arXiv:2410.13842

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), Advances in Large Margin Classifiers (pp. 61-74). MIT Press.

Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., & Sun, J. (2019). Objects365: A large-scale, high-quality dataset for object detection. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 8429-8438. https://doi.org/10.1109/ICCV.2019.00852

Yassir, A., Jai Andaloussi, S., Ouchetto, O., Mamza, K., & Serghini, M. (2023). Acoustic fish species identification using deep learning and machine learning algorithms: A systematic review. Fisheries Research, 266, 106790. https://doi.org/10.1016/j.fishres.2023.106790

Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16965-16974. https://doi.org/10.1109/CVPR52733.2024.01605
