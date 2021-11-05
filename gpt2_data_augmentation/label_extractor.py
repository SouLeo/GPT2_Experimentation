import csv
# Using readlines()
if __name__ == '__main__':
    with open('UMRF_valid_node_corrected.tsv') as csv_file:
        gold_labels = []
        csv_reader = csv.reader(csv_file, delimiter='\t')
        end_of_text_token = " <|endoftext|>"
        end_of_line = "\n"
        for row in csv_reader:
            data_str = f"{row[1]}{end_of_text_token}"
            # training_example = [f"{row[0]}", f"{row[1]}"]   # umrf lable
            gold_labels.append(data_str)

    file1 = open('predictions.txt', 'r', encoding="utf8")
    Lines = file1.readlines()

    count = 0
    # Strips the newline character
    pred = []
    for line in Lines:
        if '{' in line:
            ind = line.find('{')
            cleaned_line = line[ind:]
            pred.append(cleaned_line.rstrip())
        # print("Line{}: {}".format(count, line.strip()))

    if len(gold_labels) != len(pred):
        print('ERR')

    shifting_array = []
    pred_size = range(len(pred))
    for i in pred_size:
        gold = gold_labels[i]
        predy = pred[i]
        if gold_labels[i] == pred[i]:
            shifting_array.append(1)
        else:
            shifting_array.append(0)
    print('hi')