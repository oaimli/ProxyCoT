import os.path
import jsonlines
import json
from typing import List


def get_full_texts(sample, papers_all):
    cluster_papers = []
    for paper_id in sample["articles"]:
        paper = papers_all[paper_id]
        cluster_papers.append(paper)

    markdowns = []
    for i, paper in enumerate(cluster_papers):
        markdowns.append(paper["markdown"])

    return markdowns


def get_tables(sample, papers_all):
    cluster_papers = []
    for paper_id in sample["articles"]:
        paper = papers_all[paper_id]
        paper["article_id"] = paper_id
        cluster_papers.append(paper)

    simplified_ids = {}
    for i, paper in enumerate(cluster_papers):
        simplified_ids[paper["article_id"]] = f"article_{i}"

    # articles
    articles = []
    articles.append(["article_id", "article_title", "title_word_count", "author_count", "reference_count"])
    for paper in cluster_papers:
        article_id = simplified_ids[paper["article_id"]]
        article_title = paper["title"]
        title_word_count = len(article_title.split())
        author_count = len(paper["authors"].split(","))
        reference_count = paper["referenceCount"]
        articles.append([article_id, article_title, title_word_count, author_count, reference_count])

    # article-author
    article_author = []
    article_author.append(["article_id", "author_name", "author_position"])
    for paper in cluster_papers:
        article_id = simplified_ids[paper["article_id"]]
        paper_authors = [author.strip() for author in paper["authors"].split(",")]
        for order, author in enumerate(paper_authors):
            article_author.append([article_id, author, order])

    # citing_cited
    citing_cited = []
    citing_cited.append(["article_id_citing", "article_id_cited"])
    article_ids = []
    for paper in cluster_papers:
        article_ids.append(paper["article_id"])
    for paper in cluster_papers:
        article_id = paper["article_id"]
        paper_references = paper["references"]
        for paper_reference in paper_references:
            if paper_reference in article_ids:
                citing_cited.append([simplified_ids[article_id], simplified_ids[paper_reference]])

    return articles, article_author, citing_cited


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

    return "\n\n\n".join(articles)


def load_articles(articles_folder="../article/"):
    with open(os.path.join(articles_folder, "papers_final.json")) as f:
        articles_all = json.load(f)
    return articles_all


def load_test(prefix="64k", samples_folder="samples/final/", no_null=False, no_simple=False):
    # loading test data, one level of samples at each time
    assert prefix in ["single", "64k", "128k", "512k", "1024k"]

    samples_test = []
    save_file = os.path.join(samples_folder, f"{prefix}_samples_target.jsonl")

    with jsonlines.open(save_file) as reader:
        for line in reader:
            valid_sample = True
            if line["question"] == "":
                valid_sample = False
            if "database" in line["question"]:
                valid_sample = False
            if no_null and line["answer"] == "NULL":
                valid_sample = False
            if no_simple and line["sql_type"] == "multi_simple":
                valid_sample = False

            if valid_sample:
                if line["label"] == "test":
                    samples_test.append(line)
    print(prefix, len(samples_test))
    return samples_test


# load valid test data with full texts
def load_train(prefixes: List[str], samples_folder="samples/final/", no_null=False, no_simple=False):
    # loading training data, multiple levels of samples at each time
    assert len(set(prefixes)) == len(prefixes)
    assert  set(prefixes).issubset({"64k", "128k", "512k", "1024k"})

    samples_training = []
    samples_dev = []
    for prefix in prefixes:
        samples_training_tmp = []
        samples_dev_tmp = []
        with jsonlines.open(os.path.join(samples_folder, f"{prefix}_samples_target.jsonl")) as reader:
            for line in reader:
                valid_sample = True
                if line["question"] == "":
                    valid_sample = False
                if "database" in line["question"]:
                    valid_sample = False
                if no_null and line["answer"] == "NULL":
                    valid_sample = False
                if no_simple and line["sql_type"] == "multi_simple":
                    valid_sample = False

                if valid_sample:
                    if line["label"] == "training":
                        samples_training_tmp.append(line)

                    if line["label"] == "dev":
                        samples_dev_tmp.append(line)

        print(prefix, f"training={len(samples_training_tmp)}, dev={len(samples_dev_tmp)}")
        samples_training.extend(samples_training_tmp)
        samples_dev.extend(samples_dev_tmp)
    return samples_training, samples_dev


if __name__ == "__main__":
    # clone SciTrek and download the data in https://arxiv.org/abs/2509.21028
    # and replace the data_path with SciTrek's path
    data_path = "../../../MDRL/benchmark"

    # get samples with article ids
    train_data, dev_data = load_train(prefixes=["64k", "128k"], samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    print(len(train_data), len(dev_data))
    test_data_64 = load_test(prefix="64k", samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    test_data_128 = load_test(prefix="128k", samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    test_data_512 = load_test(prefix="512k", samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    test_data_1024 = load_test(prefix="1024k", samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    print(len(test_data_64), len(test_data_128), len(test_data_512), len(test_data_1024))

    # get full-text articles
    articles_all = load_articles(articles_folder=data_path + "/article/")

    # some statistics
    questions = set([])
    templates = set([])
    sql_queries = set([])
    for sample in train_data:
        questions.add(sample["question"])
        templates.add(sample["template"])
        sql_queries.add(sample["sql"])
    print(len(questions), len(templates), len(sql_queries))

    # load samples with full contexts, taking test_64k as example
    instruction_full = open(os.path.join("../../instructions/qwen_instruction_full_scitrek.txt")).read()
    samples_full = []
    for sample in test_data_64:
        question = sample["question"]
        answer = sample["answer"]
        markdowns = get_full_texts(sample, articles_all)
        articles = "\n\n\n".join(markdowns)
        prompt_content = instruction_full
        prompt_content = prompt_content.replace("<question>", question)
        prompt_content = prompt_content.replace("<articles>", articles)
        sample_full = {"context": prompt_content, "answer": answer}
        samples_full.append(sample_full)

    # load samples with metadata, taking train_data as example
    instruction_metadata = open(os.path.join("../../instructions/qwen_instruction_proxy_scitrek.txt")).read()
    samples_metadata = []
    for sample in train_data:
        question = sample["question"]
        answer = sample["answer"]
        articles = get_proxy(sample, articles_all)
        prompt_content = instruction_metadata
        prompt_content = prompt_content.replace("<question>", question)
        prompt_content = prompt_content.replace("<articles>", articles)
        sample_metadata = {"context": prompt_content, "answer": answer}
        samples_metadata.append(sample_metadata)