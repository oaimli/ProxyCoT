from data_loading import load_train, load_test, load_articles, get_full_texts
from datasets import load_dataset


def get_proxy(sample, papers_all):
    cluster_papers = []
    for paper_id in sample["articles"]:
        paper = papers_all[paper_id]
        paper["article_id"] = paper_id
        cluster_papers.append(paper)

    articles = []
    for paper in cluster_papers:
        article = []
        article_title = paper["title"]
        article.append(f"Article title: {article_title}")
        title_word_count = len(article_title.split())
        article.append(f"There are {title_word_count} words in the title (separated by spaces).")
        authors = paper["authors"]
        author_count = len(paper["authors"].split(","))
        article.append(f"There are {author_count} authors: {authors}")
        reference_count = paper["referenceCount"]
        if reference_count > 0:
            article.append(f"There are {reference_count} references in the reference section.")
        else:
            article.append(f"There are no reference sections.")
        papers_cited = []
        paper_references = paper["references"]
        assert len(paper_references) == len(set(paper_references))
        for paper_reference in paper_references:
            title_tmp = ""
            for paper_tmp in cluster_papers:
                if paper_reference == paper_tmp["article_id"]:
                    title_tmp = paper_tmp["title"]
                    break
            if title_tmp != "":
                papers_cited.append(title_tmp.strip())

        papers_cited = "\n".join(papers_cited)
        if len(papers_cited) > 0:
            article.append(f"The other provided articles that are cited by this article:\n{papers_cited}.")
        else:
            article.append(f"The other provided articles are not cited by this article.")
        articles.append("\n\n".join(article))

    return articles


if __name__ == "__main__":
    # clone SciTrek and download the data in https://arxiv.org/abs/2509.21028
    # and replace the dataset_dir with SciTrek's path
    dataset_dir = "../../../MDRL/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    samples_train, samples_dev = load_train(prefixes=["64k", "128k"], samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True)
    samples_test = []
    samples_test.extend(load_test(prefix="64k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    samples_test.extend(load_test(prefix="128k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))

    def process_data(samples):
        message_data = []
        for sample in samples:
            question = sample["question"]
            answer = sample["answer"]

            # metadata, use "\n\n\n" when concatenation
            metadata = get_proxy(sample, articles_all) # List(str)
            instruction_proxy = open("../../instructions/instruction_proxy_scitrek.txt").read()
            # full texts, use "\n\n\n" when concatenation
            markdowns = get_full_texts(sample, articles_all) # List(str)
            instruction_full = open("../../instructions/instruction_full_scitrek.txt").read()
            tmp = {"question": question, "answer": answer, "articles": markdowns, "instruction_full": instruction_full, "metadata": metadata, "instruction_proxy": instruction_proxy}
            message_data.append(tmp)
        return message_data

    training_data = process_data(samples_train)
    dev_data = process_data(samples_dev)
    test_data = process_data(samples_test)
    print("training/dev/test", len(training_data), len(dev_data), len(test_data))

    save_folder = "scitrek_all"
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    # with jsonlines.open(f"{save_folder}/train.jsonl", "w") as writer:
    #     writer.write_all(training_data)
    # with jsonlines.open(f"{save_folder}/val.jsonl", "w") as writer:
    #     writer.write_all(dev_data)
    # with jsonlines.open(f"{save_folder}/test.jsonl", "w") as writer:
    #     writer.write_all(test_data)

    dataset = load_dataset("json", data_files={"train": f"{save_folder}/train.jsonl", "val": f"{save_folder}/val.jsonl", "test": f"{save_folder}/test.jsonl"})
    dataset.push_to_hub("oaimli/proxycot-scitrek")


