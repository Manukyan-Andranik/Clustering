from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score as sil_coif
import argparse
from model import Model
import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run the data processing and modeling pipeline.")
    parser.add_argument("--data", type=str,
                        help="Path to the training or testing data.")
    parser.add_argument("--algo", type=str,
                        help="model type")
    parser.add_argument("--config_path", type=str,
                        help="Path to the config file.")
    args = parser.parse_args()


    if not args.data:
        print("Error: Please provide the path to the data using --data.")
    else:
        data_path = args.data
        data = pd.read_csv(data_path)
        # data, _ = make_blobs(n_samples=200, centers=3, random_state=41)
        model = Model()
        config_params = []

        if args.config_path:
            with open(args.config_path, 'r') as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        # print(config_params)

        else:
            print("Please insert values for any of these attributes (default values will be used otherwise)")
            for key in model.get_params():
                print(f"--{key}")
            print("An example of how it should be done\n--n_clusters 8 --init random")
            params_input = input()

            params_split = params_input.split()
            params_dict = {}
            for i in range(0, len(params_split) - 1, 2):
                param = params_split[i][2:]
                value = params_split[i + 1]
                for k, v in model.cluster.get_params().items():
                    if k == param:
                        value = type(v)(value)
                        break
                params_dict[param] = (value)
            config_params = params_dict

        if args.algo:
            if args.algo == "all":
                res = []
                model = [KMeans(), SpectralClustering()]
                models = ["KMeans", "Spectral"]
                for i, m in enumerate(models):
                    model[i].set_params(**config_params[m])

                    labels = model[i].fit_predict(data)
                    res.append(sil_coif(data, labels))

                    if len(data.columns) == 2:
                        fig, axs = plt.subplots()
                        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
                        fig.savefig(f"figure-{m}.png")

                res_dict = {}
                res_dict["Model"] = models
                res_dict["Silhouette score"] = res
                result = pd.DataFrame(res_dict)
                result.to_csv("result.csv", index = True)

            elif args.algo == "KMeans":
                model = Model(model=KMeans())
                model.cluster.set_params(**config_params)
                labels = model.fit_predict(data)
                print(f"Silhouette score = {sil_coif(data, labels)}")
            elif args.algo == "Spectral":
                model = Model(model=SpectralClustering())
                model.cluster.set_params(**config_params)
                labels = model.fit_predict(data)
                print(f"Silhouette score = {sil_coif(data, labels)}")
            elif args.algo == "DBSCAN":
                model = Model(model=DBSCAN())
                model.cluster.set_params(**config_params)
                labels = model.fit_predict(data)
                print(f"Silhouette score = {sil_coif(data, labels)}")


        else:
            print("Please insert values for any of these attributes (default values will be used otherwise)")
            for key in model.get_params():
                print(f"--{key}")
            print("An example of how it should be done\n--n_clusters 8 --init random")
            params_input = input()

            params_split = params_input.split()
            params_dict = {}
            for i in range(0, len(params_split) - 1, 2):
                param = params_split[i][2:]
                value = params_split[i + 1]
                for k, v in model.cluster.get_params().items():
                    if k == param:
                        value = type(v)(value)
                        break
                params_dict[param] = (value)

            model.cluster.set_params(**params_dict)

            labels = model.fit_predict(data)
            print(sil_coif(data, labels))
            if len(data.columns) == 2:
                fig, axs = plt.subplots()
                plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
                fig.savefig("figure.png")
