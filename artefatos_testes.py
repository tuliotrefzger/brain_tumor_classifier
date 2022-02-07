import pathlib
from utils_dataset import mudar_artefato, mudar_deg_level
from process_dataset import process_dataset, generate_dataloader
from neural_network import test, import_nn, define_config
import matplotlib.pyplot as plt

def test_artefato_0_10( model,
                        criterion,
                        device,
                        path_salvar_modelo,
                        test_gen,
                        func_artefato):
    acc_list = []
    loss_list = []

    # teste sem artefatos
    mudar_artefato(test_gen.dataset, None)
    accuracy, loss_test = test(model, test_gen, criterion, device, path_salvar_modelo, False)
    acc_list.append(accuracy)
    loss_list.append(loss_test)

    # teste com artefatos
    mudar_artefato(test_gen.dataset, func_artefato)
    for i in range(1, 10+1):
        mudar_deg_level(test_gen.dataset, i)
        accuracy, loss_test = test(model, test_gen, criterion, device, path_salvar_modelo, False)
        acc_list.append(accuracy)
        loss_list.append(loss_test)

    return acc_list, loss_list

def teste_artefatos_aux(model,
                    criterion,
                    device,
                    path_salvar_modelo,
                    artefatos,
                    artefatos_nomes,
                    img_size,
                    dataset_path):

    test_set = process_dataset(dataset_path,
                    img_size=img_size,
                    funcao_geradora_artefato=None,
                    nivel_degradacao=1,
                    num_classes=3,
                    augmentation=False)

    test_gen = generate_dataloader(test_set, 10, False)
    acc_list = []
    loss_list = []

    print("Iniciando teste com artefatos separados")

    for func_artefato in artefatos:
        acc, loss = test_artefato_0_10(model, criterion, device, path_salvar_modelo, test_gen, func_artefato)
        acc_list.append(acc)
        loss_list.append(loss)

    plt.figure()
    for acc, nome in zip(acc_list, artefatos_nomes):
        plt.plot(range(0, 10+1), acc, label=nome)
    plt.xticks(range(0,10+1))
    plt.yticks(range(0, 100+1, 10))
    plt.title("Accuracy(%) x Deg_Level")
    plt.xlabel("Deg_Level")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{path_salvar_modelo}accXdeglevel.png')

    plt.figure()
    for loss, nome in zip(loss_list, artefatos_nomes):
        plt.plot(range(0, 10+1), loss, label=nome)
    plt.xticks(range(0,10+1))
    plt.yticks(range(0, 8))
    plt.title("Loss x Deg_Level")
    plt.xlabel("Deg_Level")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{path_salvar_modelo}lossXdeglevel.png')
    print("Teste com artefatos separados finalizado")


# Testa cada um dos artefatos em cada um dos níveis de 0 a 10 separadamente
# no dataset de teste
def teste_artefatos(artefatos,
                    artefatos_nomes,
                    device,
                    path_salvar_modelo = './',
                    dataset_path = "./patientImages/splits",
                    img_size = 256,
                    num_classes = 3):

    # Criando pasta caso não exista
    # pathlib.Path(path_salvar_modelo).mkdir(parents=True, exist_ok=True)

    # Importando modelo. Para mudar o tipo da rede, modifique a função import_nn
    model = import_nn(num_classes, device)
    # loss function e optimizer
    criterion, _ = define_config(model, device)

    test_dataset_path = pathlib.Path(dataset_path) / "test"
    teste_artefatos_aux(model,
                    criterion,
                    device,
                    path_salvar_modelo,
                    artefatos,
                    artefatos_nomes,
                    img_size,
                    test_dataset_path)
