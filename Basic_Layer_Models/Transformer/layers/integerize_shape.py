
def integerize_shape(func):
    '''
    装饰器， 保证input_shape一定是int 或None
    :param func:
    :return:
    '''
    def convert(item):
        if hasattr(item,'__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item,'value'):
            return item.value
        else:
            return item

    def new_fuc(self,input_shape):
        input_shape = convert(input_shape)
        return func(self,input_shape)
    return new_fuc