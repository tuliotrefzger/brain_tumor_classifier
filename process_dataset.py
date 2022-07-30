import pathlib
import torch
from torchvision.datasets import ImageFolder
from utils_dataset import (
    ResizeDataset,
    aplicar_artefatos_dataset,
    BrainTumorDataset,
    aplicar_rotacao_dataset,
    get_targets,
    shuffle_pacientes,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Função pode adicionar diferentes tipos de artefatos, dar resize nas imagens
def process_dataset(
    dataset_path,
    img_size=256,
    funcao_geradora_artefato=None,
    nivel_degradacao=5,
    nivel_aleatorio_teto=10,
    nivel_aleatorio=False,
    num_classes=3,
    augmentation=False,
):

    # Carrega o dataset manualmente separado na estrutura ImageFolder do pytorch
    data_original = ImageFolder(dataset_path)

    # Resize nas imagens
    data_resize = ResizeDataset(data_original, img_size=img_size)

    # Aplicando o artefato nas imagens
    data_artefatos = aplicar_artefatos_dataset(
        data_resize,
        funcao_geradora_artefato,
        nivel_degradacao,
        nivel_aleatorio_teto,
        nivel_aleatorio,
    )

    # Augmentation
    data_aug_list = []

    # adiciona o dataset sem rotação
    data_aug_list.append(data_artefatos)

    if augmentation:
        # adiciona o dataset com outros 7 tipos de rotação
        data_aug_list.append(aplicar_rotacao_dataset(data_artefatos))

        # pode adicionar outros tipos de augmentation aqui

    # Juntando todos
    data_aug = torch.utils.data.ConcatDataset(data_aug_list)

    # Classe wrapper final
    data_final = BrainTumorDataset(data_aug, num_classes=num_classes)

    return data_final


# Função wrapper para gerar os 3 datasets com as mesmas configurações
# Espera que os dados estejam separados como:
# dataset_path/train
# dataset_path/valid
# dataset_path/test
def process_dataset_train_valid_test(
    dataset_path,
    img_size=256,
    funcao_geradora_artefato=None,
    nivel_degradacao=5,
    nivel_aleatorio_teto=10,
    nivel_aleatorio=False,
    num_classes=3,
    shuffle_pacientes_flag=False,
):

    path = pathlib.Path(dataset_path)

    if not shuffle_pacientes_flag:
        train_set = process_dataset(
            path / "train",
            img_size,
            funcao_geradora_artefato,
            nivel_degradacao,
            nivel_aleatorio_teto,
            nivel_aleatorio,
            num_classes,
        )
    else:
        train_set = process_dataset(
            path / "train",
            img_size,
            funcao_geradora_artefato,
            nivel_degradacao,
            nivel_aleatorio_teto,
            nivel_aleatorio,
            num_classes,
            augmentation=False,
        )
    valid_set = process_dataset(
        path / "valid",
        img_size,
        funcao_geradora_artefato,
        nivel_degradacao,
        nivel_aleatorio_teto,
        nivel_aleatorio,
        num_classes,
        augmentation=False,
    )
    test_set = process_dataset(
        path / "test",
        img_size,
        funcao_geradora_artefato,
        nivel_degradacao,
        nivel_aleatorio_teto,
        nivel_aleatorio,
        num_classes,
        augmentation=False,
    )

    if shuffle_pacientes_flag:
        train_set, valid_set, test_set = shuffle_pacientes(
            train_set, valid_set, test_set
        )

    return train_set, valid_set, test_set


# Gera o dataloader
# Com balancear_dataset=True, uma tentativa de balancear as classes é feita para
# tentar dar a mesma probabilidade de seleção para todas as classes.
# Esta opção pode tanto gerar undersampling quanto supersampling ao mesmo tempo
def generate_dataloader(dataset, batch_size, balancear_dataset=True):
    if not balancear_dataset:
        gen = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        )  # num_workers=0 para evitar problemas do docker # , drop_last=True
        return gen

    # Caso balancear_dataset=True
    labels_unique, counts = np.unique(get_targets(dataset), return_counts=True)
    class_weights = [sum(counts) / c for c in counts]  # Calcula o pesos
    example_weights = [
        class_weights[e] for e in get_targets(dataset)
    ]  # Precisamos criar um array com um peso para cada imagem do dataset final(pode ter mais imagens que o final)
    sampler = WeightedRandomSampler(example_weights, len(dataset))  # E cria um sampler

    gen = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        sampler=sampler,
        drop_last=True,
    )  # drop_last=True
    return gen


# É importante manter o batchsize pequeno, ou o colab vai crashar
def generate_dataloader_train_valid_test(
    train_set, valid_set, test_set, balancear_dataset=True
):
    train_batch_size = 4
    valid_batch_size = 4
    test_batch_size = 10
    train_gen = generate_dataloader(train_set, train_batch_size, balancear_dataset)
    valid_gen = generate_dataloader(valid_set, valid_batch_size, balancear_dataset)
    test_gen = generate_dataloader(test_set, test_batch_size, balancear_dataset)

    return train_gen, valid_gen, test_gen
