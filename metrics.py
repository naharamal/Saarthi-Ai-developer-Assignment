import tensorflow as tf
def f1_score_class0(labels, predictions):
    """
    To calculate f1-score for the 1st class.
    """
    prec, update_op1 = tf.compat.v1.metrics.precision_at_k(labels, predictions, 1, class_id=0)
    rec,  update_op2 = tf.compat.v1.metrics.recall_at_k(labels, predictions, 1, class_id=0)

    return {
            "f1_Score_for_class0":
                ( 2*(prec * rec) / (prec + rec) , tf.group(update_op1, update_op2) )
    }