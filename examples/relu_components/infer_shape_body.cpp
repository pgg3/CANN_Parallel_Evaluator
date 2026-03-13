const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
