# coding=utf-8
def create_converted_entity_factory():

    def create_converted_entity(ag__, ag_source_map__, ag_module__):

        def tf__load(image_name, txt_name):
            do_return = False
            retval_ = ag__.UndefinedReturnValue()
            with ag__.FunctionScope('load', 'fscope', ag__.STD) as fscope:
                image = ag__.converted_call(tf.io.read_file, (image_name,), None, fscope)
                image = ag__.converted_call(tf.image.decode_jpeg, (image,), None, fscope)
                txt = ag__.converted_call(tf.compat.v1.py_func, (my_func, [txt_name], tf.float32), None, fscope)
                txt = ag__.converted_call(tf.reshape, (txt, [257, 11, 1]), None, fscope)
                input_image = ag__.converted_call(tf.cast, (image, tf.float32), None, fscope)
                real_data = ag__.converted_call(tf.cast, (txt, tf.float32), None, fscope)
                input_image = ag__.converted_call(tf.image.random_contrast, (input_image,), dict(lower=4, upper=6), fscope)
                input_image = ag__.converted_call(tf.image.random_saturation, (input_image,), dict(lower=2, upper=4), fscope)
                input_image = ((input_image / 127.5) - 1)
                try:
                    do_return = True
                    retval_ = fscope.mark_return_value((input_image, real_data))
                except:
                    do_return = False
                    raise
            (do_return,)
            return ag__.retval(retval_)
        tf__load.ag_source_map = ag_source_map__
        tf__load.ag_module = ag_module__
        tf__load = ag__.autograph_artifact(tf__load)
        return tf__load
    return create_converted_entity