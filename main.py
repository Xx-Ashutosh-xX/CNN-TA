import cnn_model
from cnn_model import CNN
from cnn_model import get_loss_function, get_optimizer

temp = CNN()
loss_function = get_loss_function()
optimizer = get_optimizer()

# temp.build((None,15,15,1))

temp.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])

print(temp.model().summary())                