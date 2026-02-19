"""Create class reports for morph features."""
import os
import argparse
from re import search


domains = ['agriculture', 'conversational', 'entertainment',
           'judiciary', 'news-dev', 'news-testing', 'tourism']


def find_file_paths(folder_path):
    """Find file paths from a folder."""
    for root, dirs, files in os.walk(folder_path):
        return [os.path.join(root, fl) for fl in files]


def find_gold_pred_paths_and_create_reports(gold_paths, pred_paths, report_folder):
    """Create reports after finding gold and pred paths."""
    assert len(gold_paths) == len(pred_paths)
    for gold_path in gold_paths:
        gold_file_name = gold_path[gold_path.rfind('/') + 1:]
        print(gold_file_name)
        domain = search('|'.join(domains), gold_file_name).group(0)
        for pred_path in pred_paths:
            pred_file_name = pred_path[pred_path.rfind('/') + 1:]
            print(gold_file_name, pred_file_name)
            if domain in pred_file_name:
                print(gold_file_name, pred_file_name, domain)
                token_gold_pred_chunk_path = os.path.join(report_folder, domain + '-token-gold-pred-chunks.txt')
                report_path = os.path.join(report_folder, 'report-' + domain + '-chunks.txt')
                os.system('paste ' + gold_path + ' ' + pred_path + '>' + token_gold_pred_chunk_path)
                os.system('python precision_recall_score_chunking.py ' + token_gold_pred_chunk_path + ' ' + report_path)
                break


def main():
    """Pass arguments and call functions here."""
    parser = argparse.ArgumentParser(description="This a program for report generation for morph features")
    parser.add_argument('--gold', dest='g', help='Enter the folder path for gold outputs')
    parser.add_argument('--pred', dest='p', help='Enter the folder path for pred outputs')
    parser.add_argument('--report', dest='r', help='Enter the folder path where the reports will be saved')
    args = parser.parse_args()
    gold_folder = args.g
    pred_folder = args.p
    report_folder = args.r
    if not os.path.isdir(report_folder):
        os.makedirs(report_folder)
    gold_file_paths = find_file_paths(gold_folder)
    pred_file_paths = find_file_paths(pred_folder)
    find_gold_pred_paths_and_create_reports(gold_file_paths, pred_file_paths, report_folder)


if __name__ == '__main__':
    main()
