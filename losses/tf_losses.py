from tensorflow.python.ops.losses.losses_impl import *

@tf_export(v1=["losses.softmax_cross_entropy"])
def multiclass_focal_loss(
    onehot_labels, logits, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Creates a focal-loss.
  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes
  Note that `onehot_labels` and `logits` must have the same shape,
  e.g. `[batch_size, num_classes]`. The shape of loss is decided by the shape of `logits`.
  In case the shape of `logits` is `[batch_size, num_classes]`, loss is
  a `Tensor` of shape `[batch_size]`.
  Args:
    onehot_labels: One-hot-encoded labels.
    logits: Logits outputs of the network.
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has shape `[batch_size]`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`.  Also if
      `onehot_labels` or `logits` is None.
  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if onehot_labels is None:
    raise ValueError("onehot_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "focal_loss",
                      (logits, onehot_labels)) as scope:
    logits = ops.convert_to_tensor(logits)
    onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    if label_smoothing > 0:
      num_classes = math_ops.cast(
          array_ops.shape(onehot_labels)[-1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    onehot_labels = array_ops.stop_gradient(
        onehot_labels, name="labels_stop_gradient")
    losses = nn.softmax_cross_entropy_with_logits_v2(
        labels=onehot_labels, logits=logits, name="xentropy")

    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)