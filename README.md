# Глубокие генеративные модели ДЗ№2

Выполнил: Володин Антон Сергеевич

Задача: Имплементация GAN

Датасет: CelebA

Цель: Получить сходимость GAN и сгенерировать изображения человеческих лиц

## Порядок выполнения:
1. Скачать датасет CelebA

2. Имплементировать CSPup блок (5 баллов)
  
![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/6987b4be-1ed5-47b5-bf00-b8ac16b809b5)

3. Имплементировать генератор GAN по заданной архитектурной схеме (10 баллов)

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/ea83796b-8235-45a3-94f1-b7a201face95)

4. Обучить имплементированный GAN (5 баллов)

5. Добиться схдимости (регуляризации, изменение архитектуры, фишки с train loop) (10 баллов)

## Результаты
### Эксперимент 1: обучение baseline

Генератор:
```
generator =  nn.Sequential(
    nn.ConvTranspose2d(latent_dim, 512, kernel_size=2, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(negative_slope=0.2, inplace=True),
    CSPup(512),
    CSPup(256),
    CSPup(128),
    CSPup(64),
    CSPup(32),
    nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
    nn.Tanh()
)
```

Дискриминатор:
```
discriminator = nn.Sequential(
    Block(3, 64, kernel_size=3, stride=2, padding=1), 
    Block(64, 128, kernel_size=3, stride=2, padding=1), 
    Block(128, 256, kernel_size=3, stride=2, padding=1),
    Block(256, 512, kernel_size=3, stride=2, padding=1), 
    Block(512, 1024, kernel_size=3, stride=2, padding=1), 
    nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)
```

где Block - последовательное применение Conv2d, BatchNorm2d и ReLU

Параметры:
* lr = 2e-4
* epochs = 10
* batch_size = 256
* оптимизатор Adam c параметрами betas=(0.5, 0.999) для генератора и дискриминатора
* scheduler: StepLR c параметром gamma=0.9

График loss функции:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/4280ea53-030e-4423-ad1c-d42240b7ad4b)

Примеры картинок:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/0962ccfd-629d-47df-888c-0911273b668b)

Выводы: с данным набором параметров GAN разошелся, дискриминатор переиграл генератор, на картинках генерируется шум

### Эксперимент 2: использование LeakyReLU и soft меток

Идея эксперимента:
* Меняем ReLU на LeakyReLU с параметром negative_slope=0.2
* Fake и Real метки для дискриминатора заменяем на мягкие: для Fake из диапазона (0.1, 0.3), для Real из диапазона (0.7, 0.9)

Параметры:
* lr = 2e-4
* epochs = 10
* batch_size = 256
* оптимизатор Adam c параметрами betas=(0.5, 0.999) для генератора и дискриминатора
* scheduler: StepLR c параметром gamma=0.9

График loss функции:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/adf41fcc-0d95-436c-92bd-60715f173ad6)

Примеры картинок:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/d8ca2492-1ffc-4098-a4fb-4e1afd1629db)

Выводы: GAN гораздо лучше сходится по сравнению с baseline, но при этом качество картинок посредственное, а также наблюдается mode collapse

### Эксперимент 3: замедление обучения дискриминатора 
Берем за основу Эксперимент 2 с использованием LeakyReLU и soft меток

Идея эксперимента:
* будем обучать дискриминатор каждую третью итерацию

Параметры:
* lr = 2e-4
* epochs = 10
* batch_size = 256
* оптимизатор Adam c параметрами betas=(0.5, 0.999) для генератора и дискриминатора
* scheduler: StepLR c параметром gamma=0.9

График loss функции:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/967779be-3200-47e5-a62b-27b082188a66)

Примеры картинок:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/a0eb2f25-4aba-46f1-80f5-3756020e9b51)

Вывод: результат стал хуже по сравнению с Экспериментом 2, генератор переиграл дискриминатор, картинки генерируют что-то среднее между шумом и лицом человека

### Эксперимент 4: учеличение размера батча и регуляризация дискриминатора

За основу берем Эксперимент 2 с использованием LeakyReLU и soft меток

Идея эксперимента:
* увеличим размер батча с 256 до 512
* используем AdamW вместо Adam для дискриминатора

Параметры:
* lr = 2e-4
* epochs = 10
* batch_size = 512
* оптимизатор AdamW c параметрами betas=(0.5, 0.999), weight_decay=0.01 для дискриминатора, Adam c параметрами betas=(0.5, 0.999) для генератора
* scheduler: StepLR c параметром gamma=0.9

График loss функции:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/c701356b-1c57-48b6-bed5-ded215ec656c)

Примеры картинок:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/004fb414-fa25-481c-8372-532f45f36e17)

Выводы: изменения не сильно повлияли на результаты, все также качество картинок плохое и наблюдается mode collapse
