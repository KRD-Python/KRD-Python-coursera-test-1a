{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled2.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMr1cEDpriGXVUGavgcZXeO",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/KRD-Python/KRD-Python-coursera-test-1a/blob/gh-pages/TF_DOG_CAT.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0lmpEsQv6EY4",
        "outputId": "bd7c1f03-b972-43f6-c30d-19916b5c8e0a"
      },
      "source": [
        "!wget -c https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip\n",
        "!unzip -qq Cat_Dog_data.zip"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "--2021-07-21 00:48:19--  https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip\n",
            "Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.129.16\n",
            "Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.129.16|:443... connected.\n",
            "HTTP request sent, awaiting response... 200 OK\n",
            "Length: 580495262 (554M) [application/zip]\n",
            "Saving to: ‘Cat_Dog_data.zip’\n",
            "\n",
            "Cat_Dog_data.zip    100%[===================>] 553.60M  47.1MB/s    in 12s     \n",
            "\n",
            "2021-07-21 00:48:32 (44.8 MB/s) - ‘Cat_Dog_data.zip’ saved [580495262/580495262]\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iPALQGLz7UdS"
      },
      "source": [
        "import numpy as np # linear algebra\n",
        "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "plt.style.use('ggplot')\n",
        "from sklearn.model_selection import train_test_split\n",
        "import os, shutil\n",
        "from os import listdir\n",
        "from os.path import isfile, join\n",
        "import tensorflow as tf"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IenQxCjP617T",
        "outputId": "abb117ec-8acd-493b-f102-cc877ab35b58"
      },
      "source": [
        "!find $\"Cat_Dog_data\" -type d -print"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cat_Dog_data\n",
            "Cat_Dog_data/train\n",
            "Cat_Dog_data/train/cat\n",
            "Cat_Dog_data/train/dog\n",
            "Cat_Dog_data/test\n",
            "Cat_Dog_data/test/cat\n",
            "Cat_Dog_data/test/dog\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "l0EcXpey7O0D"
      },
      "source": [
        "train_val_split = len(listdir(\"Cat_Dog_data/train/cat\")) //4"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "U7EPustU76am",
        "outputId": "44dc9fb7-4ba7-4c06-fc4b-9be451409691"
      },
      "source": [
        "try:\n",
        "  os.mkdir(os.path.join(\"Cat_Dog_data\", 'validation'))\n",
        "except Exception:\n",
        "  pass\n",
        "try:\n",
        "  os.mkdir(os.path.join(\"Cat_Dog_data\", 'validation/cat'))\n",
        "except Exception:\n",
        "  pass\n",
        "try:\n",
        "  os.mkdir(os.path.join(\"Cat_Dog_data\", 'validation/dog'))\n",
        "except Exception:\n",
        "  pass\n",
        "# Print directory structure\n",
        "!find $\"Cat_Dog_data\" -type d -print"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cat_Dog_data\n",
            "Cat_Dog_data/train\n",
            "Cat_Dog_data/train/cat\n",
            "Cat_Dog_data/train/dog\n",
            "Cat_Dog_data/validation\n",
            "Cat_Dog_data/validation/cat\n",
            "Cat_Dog_data/validation/dog\n",
            "Cat_Dog_data/test\n",
            "Cat_Dog_data/test/cat\n",
            "Cat_Dog_data/test/dog\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SidKWqwS-RLe",
        "outputId": "43f82573-c2a9-4f8f-ba36-576d5e78485f"
      },
      "source": [
        "# list all files in the cat directory \n",
        "onlyfiles = [f for f in listdir(\"Cat_Dog_data/train/cat\") if isfile(join(\"Cat_Dog_data/train/cat\", f))]\n",
        "print(len(onlyfiles))\n",
        "onlyfiles[:10]"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "11250\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['cat.6491.jpg',\n",
              " 'cat.767.jpg',\n",
              " 'cat.3005.jpg',\n",
              " 'cat.156.jpg',\n",
              " 'cat.3679.jpg',\n",
              " 'cat.7169.jpg',\n",
              " 'cat.2164.jpg',\n",
              " 'cat.12100.jpg',\n",
              " 'cat.639.jpg',\n",
              " 'cat.1204.jpg']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jpo3aFDM-ZNV"
      },
      "source": [
        "# copy the first 1/4 to validation\n",
        "for f in onlyfiles[:train_val_split]:\n",
        "  shutil.move(os.path.join(\"Cat_Dog_data/train/cat\", f), 'Cat_Dog_data/validation/cat')"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7xiBD2dk-fzh",
        "outputId": "539cabe6-92bc-4e11-f776-e9d682f08594"
      },
      "source": [
        "# list all files in the dog directory \n",
        "onlyfiles = [f for f in listdir(\"Cat_Dog_data/train/dog\") if isfile(join(\"Cat_Dog_data/train/dog\", f))]\n",
        "print(len(onlyfiles))\n",
        "onlyfiles[:10]"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "11250\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['dog.5036.jpg',\n",
              " 'dog.4403.jpg',\n",
              " 'dog.5265.jpg',\n",
              " 'dog.4618.jpg',\n",
              " 'dog.106.jpg',\n",
              " 'dog.3760.jpg',\n",
              " 'dog.5595.jpg',\n",
              " 'dog.4700.jpg',\n",
              " 'dog.665.jpg',\n",
              " 'dog.6763.jpg']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hcbm3unU-r6v"
      },
      "source": [
        "# copy the first 1/4 to validation\n",
        "for f in onlyfiles[:train_val_split]:\n",
        "  shutil.move(os.path.join(\"Cat_Dog_data/train/dog\", f), 'Cat_Dog_data/validation/dog')"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tbDgN_ox-uYp"
      },
      "source": [
        "base_dir = \"Cat_Dog_data\"\n",
        "\n",
        "train_dir = os.path.join(base_dir, 'train')\n",
        "validation_dir = os.path.join(base_dir, 'validation')\n",
        "\n",
        "train_cats_dir = os.path.join(train_dir, 'cat')  # directory with our training cat pictures\n",
        "train_dogs_dir = os.path.join(train_dir, 'dog')  # directory with our training dog pictures\n",
        "validation_cats_dir = os.path.join(validation_dir, 'cat')  # directory with our validation cat pictures\n",
        "validation_dogs_dir = os.path.join(validation_dir, 'dog')  # directory with our validation dog pictures"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XOssBRUEAFdx"
      },
      "source": [
        "num_cats_tr = len(os.listdir(train_cats_dir))\n",
        "num_dogs_tr = len(os.listdir(train_dogs_dir))\n",
        "\n",
        "num_cats_val = len(os.listdir(validation_cats_dir))\n",
        "num_dogs_val = len(os.listdir(validation_dogs_dir))\n",
        "\n",
        "total_train = num_cats_tr + num_dogs_tr\n",
        "total_val = num_cats_val + num_dogs_val"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n6d5aNXtA3ky"
      },
      "source": [
        "rows = 128\n",
        "cols = 128"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wTq_BYaOCvB7",
        "outputId": "f149bf28-dfb4-4499-8109-472071f64293"
      },
      "source": [
        "model = tf.keras.models.Sequential()\n",
        "\n",
        "model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (rows, cols, 3)))\n",
        "model.add(tf.keras.layers.MaxPooling2D((2,2)))\n",
        "\n",
        "model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (rows, cols, 3)))\n",
        "model.add(tf.keras.layers.MaxPooling2D((2,2)))\n",
        "\n",
        "model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (rows, cols, 3)))\n",
        "model.add(tf.keras.layers.MaxPooling2D((2,2)))\n",
        "\n",
        "model.add(tf.keras.layers.Flatten())\n",
        "model.add(tf.keras.layers.Dense(512))\n",
        "model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))\n",
        "\n",
        "model.summary()"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "conv2d (Conv2D)              (None, 126, 126, 32)      896       \n",
            "_________________________________________________________________\n",
            "max_pooling2d (MaxPooling2D) (None, 63, 63, 32)        0         \n",
            "_________________________________________________________________\n",
            "conv2d_1 (Conv2D)            (None, 61, 61, 64)        18496     \n",
            "_________________________________________________________________\n",
            "max_pooling2d_1 (MaxPooling2 (None, 30, 30, 64)        0         \n",
            "_________________________________________________________________\n",
            "conv2d_2 (Conv2D)            (None, 28, 28, 64)        36928     \n",
            "_________________________________________________________________\n",
            "max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         \n",
            "_________________________________________________________________\n",
            "flatten (Flatten)            (None, 12544)             0         \n",
            "_________________________________________________________________\n",
            "dense (Dense)                (None, 512)               6423040   \n",
            "_________________________________________________________________\n",
            "dense_1 (Dense)              (None, 1)                 513       \n",
            "=================================================================\n",
            "Total params: 6,479,873\n",
            "Trainable params: 6,479,873\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G3UW1TdMDj17"
      },
      "source": [
        "model.compile(loss = 'binary_crossentropy', optimizer= tf.keras.optimizers.RMSprop(1e-4), metrics = ['acc'])"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "U3gBC6TJER5Q",
        "outputId": "e9e32bc0-5425-49cc-de82-fcb17c4612de"
      },
      "source": [
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "\n",
        "X_train_datagen = ImageDataGenerator(rescale = 1./255)\n",
        "X_val_datagen = ImageDataGenerator(rescale = 1./255)\n",
        "\n",
        "train_datagen = X_train_datagen.flow_from_directory(\"Cat_Dog_data/train\", target_size = (rows,cols), batch_size = 40, class_mode = 'binary')\n",
        "val_datagen = X_val_datagen.flow_from_directory(\"Cat_Dog_data/validation\", target_size = (rows,cols), batch_size = 40, class_mode = 'binary')"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 16876 images belonging to 2 classes.\n",
            "Found 5624 images belonging to 2 classes.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iW4jVAL-Gzri",
        "outputId": "01b92794-248b-4618-bab1-876ca168f108"
      },
      "source": [
        "epochs = 30\n",
        "\n",
        "H = model.fit(train_datagen, steps_per_epoch=100, epochs = epochs, validation_data=val_datagen, validation_steps=50)"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/30\n",
            "100/100 [==============================] - 49s 169ms/step - loss: 0.6944 - acc: 0.5608 - val_loss: 0.6625 - val_acc: 0.5705\n",
            "Epoch 2/30\n",
            "100/100 [==============================] - 16s 164ms/step - loss: 0.6268 - acc: 0.6482 - val_loss: 0.5987 - val_acc: 0.6670\n",
            "Epoch 3/30\n",
            "100/100 [==============================] - 16s 157ms/step - loss: 0.5776 - acc: 0.6977 - val_loss: 0.5489 - val_acc: 0.7255\n",
            "Epoch 4/30\n",
            "100/100 [==============================] - 16s 160ms/step - loss: 0.5545 - acc: 0.7085 - val_loss: 0.5888 - val_acc: 0.6770\n",
            "Epoch 5/30\n",
            "100/100 [==============================] - 17s 170ms/step - loss: 0.5356 - acc: 0.7342 - val_loss: 0.5322 - val_acc: 0.7225\n",
            "Epoch 6/30\n",
            "100/100 [==============================] - 16s 160ms/step - loss: 0.5211 - acc: 0.7523 - val_loss: 0.5190 - val_acc: 0.7380\n",
            "Epoch 7/30\n",
            "100/100 [==============================] - 16s 157ms/step - loss: 0.5093 - acc: 0.7495 - val_loss: 0.5177 - val_acc: 0.7420\n",
            "Epoch 8/30\n",
            "100/100 [==============================] - 17s 168ms/step - loss: 0.4981 - acc: 0.7527 - val_loss: 0.6004 - val_acc: 0.6900\n",
            "Epoch 9/30\n",
            "100/100 [==============================] - 16s 162ms/step - loss: 0.4763 - acc: 0.7722 - val_loss: 0.5018 - val_acc: 0.7575\n",
            "Epoch 10/30\n",
            "100/100 [==============================] - 17s 166ms/step - loss: 0.4682 - acc: 0.7697 - val_loss: 0.5185 - val_acc: 0.7495\n",
            "Epoch 11/30\n",
            "100/100 [==============================] - 16s 157ms/step - loss: 0.4461 - acc: 0.7950 - val_loss: 0.5475 - val_acc: 0.7250\n",
            "Epoch 12/30\n",
            "100/100 [==============================] - 16s 156ms/step - loss: 0.4485 - acc: 0.7968 - val_loss: 0.5666 - val_acc: 0.7265\n",
            "Epoch 13/30\n",
            "100/100 [==============================] - 17s 167ms/step - loss: 0.4234 - acc: 0.8067 - val_loss: 0.5038 - val_acc: 0.7500\n",
            "Epoch 14/30\n",
            "100/100 [==============================] - 16s 163ms/step - loss: 0.4292 - acc: 0.8058 - val_loss: 0.4492 - val_acc: 0.7830\n",
            "Epoch 15/30\n",
            "100/100 [==============================] - 16s 157ms/step - loss: 0.4220 - acc: 0.8062 - val_loss: 0.4669 - val_acc: 0.7740\n",
            "Epoch 16/30\n",
            "100/100 [==============================] - 16s 157ms/step - loss: 0.4190 - acc: 0.8115 - val_loss: 0.4590 - val_acc: 0.7885\n",
            "Epoch 17/30\n",
            "100/100 [==============================] - 16s 165ms/step - loss: 0.4170 - acc: 0.8070 - val_loss: 0.4427 - val_acc: 0.8050\n",
            "Epoch 18/30\n",
            "100/100 [==============================] - 16s 157ms/step - loss: 0.4081 - acc: 0.8192 - val_loss: 0.4445 - val_acc: 0.7940\n",
            "Epoch 19/30\n",
            "100/100 [==============================] - 16s 155ms/step - loss: 0.3997 - acc: 0.8186 - val_loss: 0.4972 - val_acc: 0.7760\n",
            "Epoch 20/30\n",
            "100/100 [==============================] - 16s 165ms/step - loss: 0.3745 - acc: 0.8358 - val_loss: 0.4488 - val_acc: 0.7910\n",
            "Epoch 21/30\n",
            "100/100 [==============================] - 16s 162ms/step - loss: 0.3904 - acc: 0.8291 - val_loss: 0.4290 - val_acc: 0.7940\n",
            "Epoch 22/30\n",
            "100/100 [==============================] - 16s 157ms/step - loss: 0.3892 - acc: 0.8292 - val_loss: 0.4684 - val_acc: 0.7805\n",
            "Epoch 23/30\n",
            "100/100 [==============================] - 16s 158ms/step - loss: 0.3666 - acc: 0.8403 - val_loss: 0.4322 - val_acc: 0.8080\n",
            "Epoch 24/30\n",
            "100/100 [==============================] - 17s 166ms/step - loss: 0.3583 - acc: 0.8395 - val_loss: 0.4438 - val_acc: 0.7930\n",
            "Epoch 25/30\n",
            "100/100 [==============================] - 16s 161ms/step - loss: 0.3616 - acc: 0.8393 - val_loss: 0.4069 - val_acc: 0.8160\n",
            "Epoch 26/30\n",
            "100/100 [==============================] - 16s 159ms/step - loss: 0.3594 - acc: 0.8445 - val_loss: 0.4644 - val_acc: 0.7800\n",
            "Epoch 27/30\n",
            "100/100 [==============================] - 17s 167ms/step - loss: 0.3435 - acc: 0.8470 - val_loss: 0.4496 - val_acc: 0.7960\n",
            "Epoch 28/30\n",
            "100/100 [==============================] - 16s 164ms/step - loss: 0.3379 - acc: 0.8518 - val_loss: 0.4293 - val_acc: 0.8080\n",
            "Epoch 29/30\n",
            "100/100 [==============================] - 17s 168ms/step - loss: 0.3228 - acc: 0.8597 - val_loss: 0.4329 - val_acc: 0.8020\n",
            "Epoch 30/30\n",
            "100/100 [==============================] - 16s 158ms/step - loss: 0.3343 - acc: 0.8533 - val_loss: 0.4232 - val_acc: 0.8080\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xP0ugwHFqa_q",
        "outputId": "f9f3427f-543d-4aff-d15b-af65f37fb320",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "H.history.keys()"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 530
        },
        "id": "DqYOb2dbJAjf",
        "outputId": "82dd5f50-ae46-481f-fcfc-e492675b314c"
      },
      "source": [
        "acc = H.history['acc']\n",
        "val_acc = H.history['val_acc']\n",
        "loss = H.history['loss']\n",
        "val_loss = H.history['val_loss']\n",
        "\n",
        "epochs=  range(1, len(acc)+1)\n",
        "\n",
        "plt.plot(epochs, acc, 'bo', label = 'Training Acc')\n",
        "plt.plot(epochs, val_acc, 'g', label = 'Val Acc')\n",
        "plt.legend()\n",
        "\n",
        "plt.figure()\n",
        "plt.plot(epochs, loss, 'bo', label = 'Training Loss')\n",
        "plt.plot(epochs, val_loss, 'g', label = 'Val Loss')\n",
        "plt.legend()\n",
        "\n"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f85ca0b9e10>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    }
  ]
}