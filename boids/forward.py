def forward(model, x, a, i, training=None):
    """Computes one forward pass of the GNCA"""
    x_pred = model([x, a, i], training=training)
    return x_pred
