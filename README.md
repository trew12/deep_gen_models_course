# Глубокие генеративные модели
## ДЗ 4. Обучение Stable diffusion 1.5 методом Dreambooth
Выполнил: Володин Антон Сергеевич

Исходный датасет из 18 фото Мэйби Бэйби: [тык](https://disk.yandex.ru/d/tWwDiOzDQvOtJA)

Использованные промпты:

```
    {
     "name": "kitchen",
     "prompt":f"close up portrait of sks woman face, in the kitchen, standing, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands disconnected limbs, mutation, ugly, blurry, amputation",
    },
    {
     "name": "forest",
     "prompt":f"portrait of sks woman face, in the forest, standing, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands disconnected limbs, mutation, ugly, blurry, amputation",
    },
    {
     "name": "street",
     "prompt":f"portrait of sks woman face, on the street, lights, midnight, NY, standing, 4K, raw, hrd, hd, high quality, realism, sharp focus,  beautiful eyes, detailed eyes",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands, mutation, ugly, blurry",
    },
    {
     "name": "beach",
     "prompt":f"mid range portrait of skswoman face, on the beach, standing, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands disconnected limbs, mutation, ugly, blurry, amputation",
    },
    {
     "name": "moon",
     "prompt":f"portrait of sks woman face, on the moon, standing, 4K, raw, hrd, hd, high quality, realism, sharp focus",
     "n_prompt":"naked, nsfw, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limb, missing limb, floating limbs, mutated hands disconnected limbs, mutation, ugly, blurry, amputation",
    },
```

### 1. Обучение Stable Diffusion c Unet весами

#### Эксперимент 1: lr = 0.000002
Примеры фотографий:

beach:
![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/56ea8437-73dd-4b60-b87a-139eff9ab95c)

forest:
![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/92cf8bec-ebda-44de-bd91-37533817ba8d)

Результат: фотографии хорошего качества, нет видимых деффектов, но результат не сильно похож на целевого персонажа

#### Эксперимент 2: lr = 0.00002
Попробуем увеличить lr на порядок 

Примеры фотографий:

street:
![street (1)](https://github.com/trew12/deep_gen_models_course/assets/64497667/96b4a7ec-0605-46ae-aca9-68d74932b96d)

forest:
![forest (1)](https://github.com/trew12/deep_gen_models_course/assets/64497667/20cd2eed-1918-437d-b241-8187d59b2b49)

Результат: результат похож на целевого персонажа, но есть визуальные деффекты и по 3-4 объекта на фотографиях

#### Эксперимент 3: lr = 0.000004
Попробуем увеличивать lr не так сильно 

Примеры фотографий:

forest:
![forest (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/b3d9cadf-91ae-4ffd-a395-e0e917b6ce2d)

beach:
![beach (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/ce791cae-1d6c-48bf-95c9-4dfdc943f9dd)

moon:
![moon (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/4c796740-3874-499f-8561-814a8ff04cc3)

street:
![street (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/bf0fcc31-95bf-420c-a361-81eadb852a2e)

kitchen:
![kitchen (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/1746f545-6e6d-444d-9730-c866112c36fe)

Результат: результат часто похож на целевого персонажа, но иногда встречаются визуальные деффекты


### 2. Обучение Lora
Чтобы не засорять большим количеством изображений результаты по ссылке: [тык](https://disk.yandex.ru/d/a3s8HDninvBunw)

#### Эксперимент 1: lr = 0.00002
Пробуем разные значения rank: 4, 8, 16, 32

Результат: персонаж вообще не похож на целевого

#### Эксперимент 2: lr = 0.0002
Пробуем разные значения rank: 4, 8, 16, 32, 64
Результат: получаемые персонажи слишком однотипные, а генерация низкого качества

#### Эксперимент 3: lr = 0.0001
Пробуем разные значения rank: 8, 16, 32
Результат: персонаж часто похож на целевого, картинки довольно неплохого качества, но все равно хуже чем Unet

С повышением rank начинают генерироваться картинки более хорошего качества, но увеличивается время

### 3. Сравнение Unet и Lora
Результаты для Unet (lr=0.000004):

forest:
![forest (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/b3d9cadf-91ae-4ffd-a395-e0e917b6ce2d)

beach:
![beach (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/ce791cae-1d6c-48bf-95c9-4dfdc943f9dd)

moon:
![moon (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/4c796740-3874-499f-8561-814a8ff04cc3)

street:
![street (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/bf0fcc31-95bf-420c-a361-81eadb852a2e)

kitchen:
![kitchen (2)](https://github.com/trew12/deep_gen_models_course/assets/64497667/1746f545-6e6d-444d-9730-c866112c36fe)

Результаты для Lora (exp3: lr=0.0001, rank=32):

forest:
![forest](https://github.com/trew12/deep_gen_models_course/assets/64497667/16bda74b-30b3-4a60-8ccd-3199e46100cf)

beach:
![beach](https://github.com/trew12/deep_gen_models_course/assets/64497667/19428b29-ab3a-4de1-910f-6554de07909f)

moon:
![moon](https://github.com/trew12/deep_gen_models_course/assets/64497667/03d1a1a5-106e-4a3f-aa90-972c56becb6b)

street:
![street](https://github.com/trew12/deep_gen_models_course/assets/64497667/7718e027-c7cd-423e-9794-9e10ac3a8d36)

kitchen:
![kitchen](https://github.com/trew12/deep_gen_models_course/assets/64497667/c7257741-cc61-4ab8-ab6f-cb4a4b349165)

Вывод: видим, что генерации с Unet боллее хорошие, персонаж часто похож на себя, а у Lora генерации довольно часто мыльные

### 4. ControlNet
Используем вариант с Canny

Как референс используем картину "Девушка с жемчужной серёжкой":

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/de670026-d7e2-485a-9785-0b06045d4fd1)

Границы:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/91172fa3-1982-4222-a359-8b9e037cfb6a)

Результат:

![image](https://github.com/trew12/deep_gen_models_course/assets/64497667/43735f6c-0c3e-4fd5-a58b-60b66ddb2f74)


