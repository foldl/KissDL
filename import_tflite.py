import flatbuffers
from os.path import dirname, basename, isfile, join
import glob, sys
import tflite
import importlib

# load all tflite modules
modules = glob.glob(join(dirname(getattr(sys.modules['tflite'], '__file__', None)), "*.py"))
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        importlib.import_module('tflite.' + basename(f)[:-3])

op_code_dict = {}
activate_fn_dict = {}

def InitBuiltinCodeDict():
    o = tflite.BuiltinOperator.BuiltinOperator()
    map = dir(o)
    for x in map:
        if not x.startswith('_'):
            op_code_dict[o.__getattribute__(x)] = "UNSUPPORTED " + x
    op_code_dict[o.CONV_2D] = 'conv_2d'
    op_code_dict[o.DEPTHWISE_CONV_2D] = 'depthwise_conv_2d'
    op_code_dict[o.MAX_POOL_2D] = 'max_pool_2d'
    op_code_dict[o.FULLY_CONNECTED] = 'dense'
    op_code_dict[o.SOFTMAX] = 'softmax'

    o = tflite.ActivationFunctionType.ActivationFunctionType()
    activate_fn_dict[o.NONE] = 'none'
    activate_fn_dict[o.RELU] = 'relu'
    activate_fn_dict[o.RELU_N1_TO_1] = 'relu1'
    activate_fn_dict[o.RELU6] = 'relu6'
    activate_fn_dict[o.TANH] = 'tanh'
    activate_fn_dict[o.SIGN_BIT] = 'sign'


def formatType(f):
    if f == tflite.TensorType.TensorType.FLOAT32:
        return 'float32'
    if f == tflite.TensorType.TensorType.FLOAT16:
        return 'float16'
    if f == tflite.TensorType.TensorType.INT32:
        return 'int32'
    if f == tflite.TensorType.TensorType.UINT8:
        return 'uint8'
    if f == tflite.TensorType.TensorType.INT64:
        return 'int64'
    if f == tflite.TensorType.TensorType.STRING:
        return 'str'
    if f == tflite.TensorType.TensorType.BOOL:
        return 'bool'
    if f == tflite.TensorType.TensorType.INT16:
        return 'int16'
    if f == tflite.TensorType.TensorType.COMPLEX64:
        return 'complex64'
    raise Exception("unknown type")

def getShape(t: tflite.Tensor.Tensor):
    r = []
    for i in range(0, t.ShapeLength()):
        r.append(t.Shape(i))
    return r

def format_quantization(q: tflite.QuantizationParameters.QuantizationParameters):
    items = []
    if q.Details() == None:
        if q.MinLength() > 0:
            items.append('{min, %s}' % (q.MinAsNumpy().tolist()))
        if q.MaxLength() > 0:
            items.append('{max, %s}' % (q.MaxAsNumpy().tolist()))
        if q.ScaleLength() > 0:
            items.append('{scale, %s}' % (q.ScaleAsNumpy().tolist()))
        if q.ZeroPointLength() > 0:
            items.append('{zero_pt, %s}' % (q.ZeroPointAsNumpy().tolist()))
    else:
        raise Exception('unsupported QuantizationDetails')

    if q.QuantizedDimension() != 0:
        items.append('{dimention, %s}' % (q.QuantizedDimension()))

    return '{quantization, [' + ', '.join(items) + ']}'

def get_buffer(m: tflite.Model.Model, i):
    b = m.Buffers(i)
    d = ''
    if b.DataLength() > 0:
        d = str(b.DataAsNumpy().tolist())[1:-1]
    return d

def convert_tensor(m, tensor: tflite.Tensor.Tensor, fpath, id):
    s = '{tensor, "%s", %s, %s, ' % (
        str(tensor.Name(), encoding='utf8'),
        formatType(tensor.Type()),
        getShape(tensor)
        )

    props = []
    if tensor.Quantization() != None:
        props.append(format_quantization(tensor.Quantization()))
    if tensor.Sparsity() != None:
        raise Exception('Sparsity is unsupported')

    props.append('{is_var, %s}' % (str(tensor.IsVariable()).lower()))
    data = get_buffer(m, tensor.Buffer())
    if data != '':
        with open(join(fpath, '%s.dat' % id), 'w') as f:
            f.write("<<" + data + ">>.")
        props.append('{data, %s}' % id)

    return s + '[' + ', '.join(props) + ']}.'

def parse_conv2d_opt(opt: tflite.Conv2DOptions.Conv2DOptions, props: list):
    props.append('{padding, %s}' % "same" if opt.Padding() == tflite.Padding.Padding.SAME else "zero")
    props.append('{stride, {%s, %s}}' % (opt.StrideW(), opt.StrideH()))
    props.append('{activation, %s}' % activate_fn_dict[opt.FusedActivationFunction()])
    props.append('{dilation_factor, {%s, %s}}' % (opt.DilationWFactor(), opt.DilationHFactor()))

def parse_depth2d_opt(opt, props):
    props.append('{padding, %s}' % "same" if opt.Padding() == tflite.Padding.Padding.SAME else "zero")
    props.append('{stride, {%s, %s}}' % (opt.StrideW(), opt.StrideH()))
    props.append('{depth_multiplier, %s}' % opt.DepthMultiplier())
    props.append('{activation, %s}' % activate_fn_dict[opt.FusedActivationFunction()])
    props.append('{dilation_factor, {%s, %s}}' % (opt.DilationWFactor(), opt.DilationHFactor()))

def parse_dense_opt(opt, props):
    props.append('{activation, %s}' % activate_fn_dict[opt.FusedActivationFunction()])
    if opt.WeightsFormat() != tflite.FullyConnectedOptionsWeightsFormat.FullyConnectedOptionsWeightsFormat.DEFAULT:
        raise "FullyConnectedOptionsWeightsFormat is not supported"
    if opt.KeepNumDims():
        raise "KeepNumDims is not supported"
    #props.append('{keep_num_dims, %s}' % opt.KeepNumDims())

def parse_pool_2d_opt(opt, props):
    props.append('{padding, %s}' % "same" if opt.Padding() == tflite.Padding.Padding.SAME else "zero")
    props.append('{stride, {%s, %s}}' % (opt.StrideW(), opt.StrideH()))
    props.append('{filter, {%s, %s}}' % (opt.FilterWidth(), opt.FilterHeight()))
    props.append('{activation, %s}' % activate_fn_dict[opt.FusedActivationFunction()])

def convert_op(model: tflite.Model.Model, g: tflite.SubGraph.SubGraph, o: tflite.Operator.Operator):
    code = model.OperatorCodes(o.OpcodeIndex()).BuiltinCode()
    inputs = []
    for i in range(0, o.InputsLength()):
        inputs.append(o.Inputs(i))
    outputs = []
    for i in range(0, o.OutputsLength()):
        outputs.append(o.Outputs(i))
    s = "{op, '%s', [%s], [%s], [" % (op_code_dict[code], get_tensors_name(g, inputs), get_tensors_name(g, outputs))
    props = []
    ot = o.BuiltinOptions()
    if o.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.Conv2DOptions:
        opt = tflite.Conv2DOptions.Conv2DOptions()
        opt.Init(ot.Bytes, ot.Pos)
        parse_conv2d_opt(opt, props)
    if o.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.DepthwiseConv2DOptions:
        opt = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptions()
        opt.Init(ot.Bytes, ot.Pos)
        parse_depth2d_opt(opt, props)
    elif o.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.FullyConnectedOptions:
        opt = tflite.FullyConnectedOptions.FullyConnectedOptions()
        opt.Init(ot.Bytes, ot.Pos)
        parse_dense_opt(opt, props)
    elif o.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.SoftmaxOptions:
        opt = tflite.SoftmaxOptions.SoftmaxOptions()
        opt.Init(ot.Bytes, ot.Pos)
        props.append('{beta, %s}' % opt.Beta())
    elif o.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.Pool2DOptions:
        opt = tflite.Pool2DOptions.Pool2DOptions()
        opt.Init(ot.Bytes, ot.Pos)
        parse_pool_2d_opt(opt, props)

    if o.CustomOptionsLength() > 0:
        raise Exception("CustomOptions is not supported")

    if o.MutatingVariableInputsLength() > 0:
        bs = []
        for i in range(0, o.MutatingVariableInputsLength()):
            bs.append('true' if o.MutatingVariableInputs(i) else 'false')
        props.append('{mutating_inputs, [%s]}' % ', '.join(bs))

    if o.IntermediatesLength() > 0:
        props.append('{intermediates, %s}' % get_tensors_name(o.IntermediatesAsNumpy().tolist()))

    return s + ', '.join(props) + ']}.'

def get_tensors_name(g: tflite.SubGraph.SubGraph, l):
    return ','.join(['"%s"' % str(g.Tensors(i).Name(), encoding='utf8') for i in l])

def convert(fn, kmfn, n = 0):
    buf = open(fn, 'rb').read()
    model = tflite.Model.Model.GetRootAsModel(buf, 0)
    print("Description: " + str(model.Description(), encoding='utf8'))
    print("---- Used operations -----")
    fo = open(kmfn, 'w')
    for i in range(0, model.OperatorCodesLength()):
        code = model.OperatorCodes(i).BuiltinCode()
        print(op_code_dict[code])
    print("converting ...")

    g = model.Subgraphs(n)
    fo.write("[{inputs, [%s]}, {outputs, [%s]}].\n" % (get_tensors_name(g, g.InputsAsNumpy().tolist()),
                                  get_tensors_name(g, g.OutputsAsNumpy().tolist())))

    for i in range(0, g.OperatorsLength()):
        s = convert_op(model, g, g.Operators(i))
        fo.write(s + '\n')

    for i in range(0, g.TensorsLength()):
        s = convert_tensor(model, g.Tensors(i), dirname(kmfn), i)
        fo.write(s + '\n')

    print("OK")

InitBuiltinCodeDict()

#convert('./examples/tiny_conv_graph.tflite', './examples/tiny_conv_graph.emodel')

if len(sys.argv) != 3:
    print("usage: python import_tflite.py tflite_file_name result_file_name")
else:
    convert(sys.argv[1], sys.argv[2])