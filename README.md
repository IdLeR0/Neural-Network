# Нейросеть с нуля
В этом проекте реализованы все основные компоненты нейросети на языке c++.
## Функции активации
Проект поддерживает 5 функций активации:
1. ```ActivationFunc::Name::Id``` - Id
2. ```ActivationFunc::Name::ReLU``` - ReLU
3. ```ActivationFunc::Name::Softmax``` - Softmax
4. ```ActivationFunc::Name::Sigmoid``` - Sigmoid
5. ```ActivationFunc::Name::Tanh``` - Tanh
## Функции потерь
Проект поддерживает 3 функции потерь:
1. ```LossFunc::Name::Mae``` - Mean Absolute Error.
2. ```LossFunc::Name::Mse``` - Mean Squared Error.
3. ```LossFunc::Name::CrossEntropy``` - Cross‑Entropy.
## Можно генерировать случайные параметры слоев самому
Для этого необходимо создать объект класса ```RandomParams```
1. ```GenerateNormalMatrix``` - генерирует матрицу с нормальным распределением (по умолчанию ~ N(0, 1));
2. ```GenerateUniformMatrix``` - генерирует матрицу с равномерным распределением(по умолчанию с Uni(0, 1));
3. ```GenerateConstantMatrix``` - генерирует матрицу, все элементы которой заполнены одним и тем же числом (по умолчанию все элементы равны 0);
Есть также соответствующие аналоги для векторов (например,```GenerateNormalVector```).
## FileReader/FileWriter - отвечают за запись и чтение нейросети с бинарного файла (см. пример ниже)

## Процесс использование библиотеки
Взаимодействия пользователя с нейросетью можно разделить на 4 части: 
построение нейросети, подготовка данных, обучение и тестирование.
```cpp
#include <iostream>
#include "src/net.h"
using namespace network;

int main() {
    // создание нейросети из 2х слоев
    std::vector<Index> layer_sizes{784, 256, 10};

    // создаем свои параметры для слоев (необязательно)
    RandomParams rnd(42);
    LayerParams first_layer;
    LayerParams second_layer;
    first_layer.weights = rnd.GenerateNormalMatrix(256, 784, 2, 10);  // ~N(2, 10);
    first_layer.bias = rnd.GenerateConstantVector(256, 0.01);
    second_layer.weights = rnd.GenerateUniformMatrix(10, 256, -1, 1);  // ~Uni(-1, 1);
    second_layer.bias = rnd.GenerateNormalVector(10);                  // ~ N(0, 1);
    std::vector<LayerParams> layer_params = {first_layer, second_layer};

    std::vector<ActivationFunc::Name> activation_functions{
        ActivationFunc::Name::ReLU, ActivationFunc::Name::Softmax};  // Функции активации на слоях
    Net my_model(layer_sizes, activation_functions, layer_params);

    // подготовка данных
    Data train_data;
    Data test_data;
    // чтение базы данных
    // ...

    DataLoader data_loader(std::move(train_data));
    LossFunc::Name loss_func = LossFunc::Name::CrossEntropy;
    Index batch_size = 70;
    Index num_epochs = 17;

    // Параметры для алгоритма оптимизации Адам
    DataType learning_rate = 3e-4;
    DataType beta1 = 0.9;
    DataType beta2 = 0.99;
    Info print_info = Info::On;  // хотим, чтобы при обучении выводилась информация в терминал

    // обучение
    my_model.Train(data_loader, loss_func, batch_size, num_epochs, print_info, learning_rate, beta1,
                   beta2);

    // Вычисляем результат на тестовой выборки
    Matrix prediction = my_model.Evaluate(test_data.input);

    // сравниваем prediction по какому-то правилу с test.output
    // ...

    //////////////////////////////////////////////////////////////////

    //хотим сохранить нейросеть
    FileWriter w("net.bin");
    w << my_model;

    // хотим выгрузить нейросеть
    Net read_model;
    FileReader r("net.bin");
    r >> read_model;
    // read_model == my_model
}

```








