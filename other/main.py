import transformers
import datasets
import shap
from shap.plots import *
import csv


if __name__ == "__main__":
    file1 = open('WFiIS-MIO-main/datasets/realdonaldtrump.csv', 'r')
    file2 = open('WFiIS-MIO-main/datasets/trumptweets.csv', 'r')

    with open('WFiIS-MIO-main/datasets/realdonaldtrump.csv', mode='r', encoding="utf8") as infile:
        reader = csv.reader(infile)
        with open('WFiIS-MIO-main/datasets/real.csv', mode='w', encoding="utf8") as outfile:
            writer = csv.writer(outfile)
            short_data = [rows[2] for rows in reader]

    # for a in short_data:
    #     print(short_data[a])
    # short_data = [v[:500] for v in short_data[2][:20]]
    print(short_data)

    # dataset = datasets.load_dataset("imdb", split="test")
    # print(type(dataset['text']))

    # shorten the strings to fit into the pipeline model
    # short_data = [v[:500] for v in dataset[:20]]
    # classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)
    # classifier(short_data[:2])
    # # define the explainer
    # explainer = shap.Explainer(classifier)
    # # explain the predictions of the pipeline on the first two samples
    # shap_values = explainer(short_data[:2])
    # shap.plots.text(shap_values[:, :, "POSITIVE"])
