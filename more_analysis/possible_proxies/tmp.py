import sys
sys.path.append("../../")
from preparation.scitrek.data_loading import load_articles


if __name__ == "__main__":
    dataset_dir = "../../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")

    print(len(articles_all))
    invalid_articles = []
    for article_id, paper in articles_all.items():
        markdown = paper["markdown"].lower()
        reference_index = markdown.rfind("reference")
        appendix_index = markdown.rfind("appendix")
        if appendix_index == -1:
            if reference_index != -1:
                reference_section = markdown[reference_index:]
            else:
                reference_section = ""
        else:
            if reference_index != -1:
                if appendix_index > reference_index:
                    reference_section = markdown[reference_index:appendix_index]
                else:
                    reference_section = markdown[reference_index:]
            else:
                reference_section = ""

        if reference_section == "":
            invalid_articles.append(article_id)
        else:
            print("Reference:\n", reference_section)

    print(len(invalid_articles))