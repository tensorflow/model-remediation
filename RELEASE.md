<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug fixes and other small changes
* `build_min_diff_dataset` now supports SparseTensors.


## Breaking changes

## Deprecations

# Version 0.1.5

## Major Features and Improvements

* Add support for multiple applications of MinDiff within MinDiffModel. This
includes:

  *  Utils for validating and manipulating structures for MinDiff applications
  * Added functionality in input utils to enable packing multiple sets of
  MinDiff data into one dataset.
  * Changes to MinDiffModel to support multiple applications of MinDiff in a
  single instance.

## Bug fixes and other small changes
* Remove protobuf and tensorflow_model_analysis dependencies.
* `pack_min_diff_data` now conserves shape or original inputs.

## Breaking changes
* (Minor) Change default name of MinDiffLoss instance to be snake_case.

## Deprecations

# Version 0.1.4

## Major Features and Improvements

## Bug fixes and other small changes

* Add support for MinDiffModel subclasses that are also subclasses of TF
Functional class.
* Add uci related utils for colabs. This is unrelated to the actual
package api but is used in our tutorials and guides.

## Breaking changes

## Deprecations

# Version 0.1.3

## Major Features and Improvements

## Bug fixes and other small changes

* Implement `get_config` and `from_config` methods for losses, kernels and
MinDiffModel register classes as keras objects.

## Breaking changes

* (Minor) Change output of `test_step` to not include `min_diff_loss` metric if
`min_diff_data` is not included.
* (Minor) Ensure that MinDiffModel metric has a unique name. In the case of a
collision, the name will be changed to "min_diff_loss_N" for the lowest N
(starting at 1) that makes it unique.

## Deprecations

# Version 0.1.2

## Major Features and Improvements

*   Initial Model Remediation Release.

## Bug fixes and other changes

*  N/A

## Breaking changes

*   N/A

## Deprecations

*   N/A
