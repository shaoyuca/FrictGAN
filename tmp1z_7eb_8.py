# coding=utf-8
from __future__ import absolute_import, division, print_function
def create_converted_entity_factory():

    def create_converted_entity(ag__, ag_source_map__, ag_module__):

        def tf___defun_call(self, inputs):
            'Wraps the op creation method in an Eager function for `run_eagerly`.'
            do_return = False
            retval_ = ag__.UndefinedReturnValue()
            with ag__.FunctionScope('_defun_call', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                try:
                    do_return = True
                    retval_ = fscope.mark_return_value(ag__.converted_call(self._make_op, (inputs,), None, fscope))
                except:
                    do_return = False
                    raise
            (do_return,)
            return ag__.retval(retval_)
        tf___defun_call.ag_source_map = ag_source_map__
        tf___defun_call.ag_module = ag_module__
        tf___defun_call = ag__.autograph_artifact(tf___defun_call)
        return tf___defun_call
    return create_converted_entity