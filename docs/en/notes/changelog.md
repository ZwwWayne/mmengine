# Changelog

## v0.1.0 (31/8/2022)

We are excited to announce the release of MMEngine, OpenMMLab foundational library for training deep learning models, one of the core dependencies of OpenMMLab 2.0 projects.
MMEngine 0.1.0 is the first version of MMEngine, which defines the base classes with unified interfaces of the dataset, models, evaluation, and visualization in OpenMMLab 2.0 projects.

### Highlights

1. **New engine**. MMEngine is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a universal and powerful runner that allows more flexible customizations and significantly simplifies the entry points of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMEngine unifies and refactors the interfaces and internal logic of training, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logic to allow the emergence of multi-task/modality algorithms.

3. **Higher Efficiency**. We optimize the training and inference speed for common models and configurations, achieving a faster or similar speed than [Detection2](https://github.com/facebookresearch/detectron2/). Model details of benchmark will be updated in [this note](./benchmark.md#comparison-with-detectron2).

4. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmdetection.readthedocs.io/en/3.x/).

### Breaking Changes

MMEngine has undergone significant changes for better design, higher efficiency, more flexibility, and more unified interfaces.
Besides the changes in API, we briefly list the major breaking changes in this section.
We will update the [migration guide](../migration/) to provide complete details and migration instructions.
Users can also refer to the [API doc](https://mmdetection.readthedocs.io/en/3.x/) for more details.

#### Dependencies

- MMEngine runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMEngine is not guaranteed.
- MMEngine relies on MMEngine to run. MMEngine is a new foundational library for training deep learning models of OpenMMLab and is the core dependency of OpenMMLab 2.0 projects. The dependencies of file IO and training are migrated from MMCV 1.x to MMEngine.
- MMEngine relies on MMCV>=2.0.0rc0. Although MMCV no longer maintains the training functionalities since 2.0.0rc0, MMEngine relies on the data transforms, CUDA operators, and image processing interfaces in MMCV. Note that the package `mmcv` is the version that provides pre-built CUDA operators and `mmcv-lite` does not since MMCV 2.0.0rc0, while `mmcv-full` has been deprecated since 2.0.0rc0.

#### Training and testing

- MMEngine uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of the dataset, model, evaluation, and visualizer. Therefore, MMEngine no longer maintains the building logic of those modules in `mmdet.train.apis` and `tools/train.py`. Those codes have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic to that in training scripts to build the runner.
- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.
- Learning rate and momentum schedules have been migrated from Hook to [Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html). Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structure to ease the understanding of the components in the runner. Users can read the [config example of MMEngine](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects. The names of checkpoints are not updated for now as there is no BC-breaking of model weights between MMEngine and 2.x. We will progressively replace all the model weights with those trained in MMEngine. Please refer to the [user guides of config](../user_guides/config.md) for more details.

#### Dataset

The Dataset classes implemented in MMEngine all inherit from the `BaseDetDataset`, which inherits from the [BaseDataset in MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). In addition to the changes in interfaces, there are several changes in Dataset in MMEngine.

- All the datasets support serializing the internal data list to reduce the memory when multiple workers are built for data loading.
- The internal data structure in the dataset is changed to be self-contained (without losing information like class names in MMDet 2.x) while keeping simplicity.
- The evaluation functionality of each dataset has been removed from the dataset so that some specific evaluation metrics like COCO AP can be used to evaluate the prediction on other datasets.

#### Data Transforms

The data transforms in MMEngine all inherits from `BaseTransform` in MMCV>=2.0.0rc0, which defines a new convention in OpenMMLab 2.0 projects.
Besides the interface changes, there are several changes listed below:

- The functionality of some data transforms (e.g., `Resize`) are decomposed into several transforms to simplify and clarify the usages.
- The format of data dict processed by each data transform is changed according to the new data structure of dataset.
- Some inefficient data transforms (e.g., normalization and padding) are moved into data preprocessor of model to improve data loading and training speed.
- The same data transforms in different OpenMMLab 2.0 libraries have the same augmentation implementation and the logic given the same arguments, i.e., `Resize` in MMEngine and MMSeg 1.x will resize the image in the exact same manner given the same arguments.

#### Model

The models in MMEngine all inherit from `BaseModel` in MMEngine, which defines a new convention of models in OpenMMLab 2.0 projects.
Users can refer to [the tutorial of the model in MMengine](https://mmengine.readthedocs.io/en/latest/tutorials/model.html) for more details.
Accordingly, there are several changes as the following:

- The model interfaces, including the input and output formats, are significantly simplified and unified following the new convention in MMEngine.
  Specifically, all the input data in training and testing are packed into `inputs` and `data_samples`, where `inputs` contains model inputs like a list of image tensors, and `data_samples` contains other information of the current data sample such as ground truths, region proposals, and model predictions. In this way, different tasks in MMEngine can share the same input arguments, which makes the models more general and suitable for multi-task learning and some flexible training paradigms like semi-supervised learning.
- The model has a data preprocessor module, which is used to pre-process the input data of the model. In MMEngine, the data preprocessor usually does the necessary steps to form the input images into a batch, such as padding. It can also serve as a place for some special data augmentations or more efficient data transformations like normalization.
- The internal logic of the model has been changed. In MMdet 2.x, model uses `forward_train`, `forward_test`, `simple_test`, and `aug_test` to deal with different model forward logics. In MMEngine and OpenMMLab 2.0, the forward function has three modes: 'loss', 'predict', and 'tensor' for training, inference, and tracing or other purposes, respectively.
  The forward function calls `self.loss`, `self.predict`, and `self._forward` given the modes 'loss', 'predict', and 'tensor', respectively.

#### Evaluation

The evaluation in MMDet 2.x strictly binds with the dataset. In contrast, MMEngine decomposes the evaluation from dataset so that all the detection datasets can evaluate with COCO AP and other metrics implemented in MMEngine.
MMEngine mainly implements corresponding metrics for each dataset, which are manipulated by [Evaluator](https://mmengine.readthedocs.io/en/latest/design/evaluator.html) to complete the evaluation.
Users can build an evaluator in MMEngine to conduct offline evaluation, i.e., evaluate predictions that may not produce in MMEngine with the dataset as long as the dataset and the prediction follow the dataset conventions. More details can be found in the [tutorial in mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html).

#### Visualization

In OpenMMLab 2.0 projects, we use [Visualizer](https://mmengine.readthedocs.io/en/latest/design/visualization.html) to visualize data. MMEngine implements `DetLocalVisualizer` to allow visualization of ground truths, model predictions, feature maps, etc., at any place. It also supports sending the visualization data to any external visualization backends such as Tensorboard.

### Improvements

- Support mixed precision training of all the models. However, some models may get undesirable performance due to some numerical issues. We will update the documentation and list the results (accuracy of failure) of mixed precision training.

### Bug Fixes

### New Features

1.

### Planned changes

We list several planned changes of MMEngine 0.1.0 so that the community could more comprehensively know the progress of MMEngine. Feel free to create a PR, issue, or discussion if you are interested, have any suggestions and feedback, or want to participate.

1. **[Refactor]: Simplified and unified file I/O interfaces.** the current file I/O interfaces in MMEngine are directly migrated from [MMCV 1.x](https://github.com/open-mmlab/mmcv) in OpenMMLab 1.0 projects. We plan to provide a set of more simplified interfaces without breaking the compatibility. Therefore, downstream projects can migrate to the new interfaces or still use the existing interfaces without changes.
2. **[Feature]: Test-time augmentation (TTA).** TTA is supported in many OpenMMLab 1.0 projects like [MMClassification](https://github.com/open-mmlab/mmclassification), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). It is not implemented in OpenMMLab 2.0 projects due to the limited time slot. We will define the base classes and unified interfaces of TTA in MMEngine, so that downstream projects could support it in the following releases with a new and simplified design.
3. **[Feature]: Unified high-level interfaces that can be used in Jupyter Notebook or Colab.** MEngine already unifies the training, test, and validation interfaces by runner. However, there are still interfaces like inference, config printing, dataset visualization. We plan to design a unified and simplified interfaces for more functionalities so that they can be used in Jupyter Notebook, Colab, and downstream libraries.
4. **[Enhancement]: Documentation.** We will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate in the future development, and smoothly migrate to MMEngine.
5. **[Feature]: Model profiling.** MMEngine does not support some model profiling tools like FLOPs calculation for now. We plan to design a more elegant and complete manner for model profiling.
6. **[Refactor]: Weight initialization.** We continue to use the implementation of weight initialization from [MMCV 1.x](https://github.com/open-mmlab/mmcv) in OpenMMLab 1.0 projects. We will refactor the internal implementation of weight initialization without breaking the user interfaces.
7. **[Refactor]: Configuration and registry system to simplify cross-library usages.** MMEngine already supports cross-library usages of configs and registries. However, the current design introduces more conventions which makes the usages of configs more complex. We are rethinking the current design and gathering user feedbacks so that we can simplify the system in the future.

### Contributors

A total of 28 developers contributed to this release.
Thanks @HAOCHENYE, @zhouzaida, @RangiLyu, @Harold-lkk, @hhaAndroid, @YuanLiuuuuuu, @teamwong111, @imabackstabber, @ly015, @GT9505, @mzr1996, @plyfager, @jbwang1997, @ice-tong, @gaotongxiao, @vansin, @MeowZheng, @LeoXing1996, @hukkai, @ZCMax, @chhluo, @Dai-Wenxun, @VVsssssk, @C1rN09, @274869388, @fangyixiao18, @ZwwWayne, @hellock
