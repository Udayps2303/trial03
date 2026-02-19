"""Create labeled data for morph features."""
from sys import argv
from re import search


hindi_numbers = '[\U00000967-\U0000096F]+'
english_numbers = '[\U00000031-\U00000039]+'


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def process_lines_and_create_labeled_data(lines):
    """Process lines and create labeled data."""
    lcat, gender, number, person, case, vibhakti = [''] * 6
    for line in lines:
        line = line.strip()
        if line:
            token, pos, morph = line.split('\t')
            fields = morph.split(',')
            assert len(fields) == 8
            lcat = token + '\tunk' if not fields[1] else token + '\t' + fields[1]
            gender = token + '\tunk' if not fields[2] else token + '\t' + fields[2]
            number = token + '\tunk' if not fields[3] else token + '\t' + fields[3]
            person = token + '\tunk' if not fields[4] else token + '\t' + fields[4]
            case = token + '\tunk' if not fields[5] else token + '\t' + fields[5]
            if not fields[6]:
                vibhakti = token + '\tunk'
            else:
                number_match = search('(' + english_numbers + '|' + hindi_numbers + ')', fields[6])
                if not number_match:
                    vibhakti = token + '\t' + fields[6]
                else:
                    vibhakti = token + '\t' + fields[6][: number_match.start()]
            all_feat = token + '\t' + '\t'.join([lcat.split()[1], gender.split()[1], number.split()[1], person.split()[1], case.split()[1], vibhakti.split()[1]])
            yield lcat, gender, number, person, case, vibhakti, all_feat
            # vibhakti = vibhakti + [token + '\tunk'] if not fields[6] else vibhakti + [token + '\t' + fields[6]]
            # for index, field in enumerate(fields):
            #     print(index, field)
            #     if not field:
            #         field = 'unk'
            #         print(index, field)
            #     if index == 1:
            #         lcat.append(token + '\t' + field)
            #         print(lcat[-1], 'L')
            #         continue
            #     elif index == 2:
            #         gender.append(token + '\t' + field)
            #         print(gender[-1], 'G')
            #         continue
            #     elif index == 3:
            #         number.append(token + '\t' + field)
            #         continue
            #     elif index == 4:
            #         person.append(token + '\t' + field)
            #         continue
            #     elif index == 5:
            #         case.append(token + '\t' + field)
            #         continue
            #     elif index == 6:
            #         vibhakti.append(token + '\t' + field)
            #         continue
        else:
            yield '', '', '', '', '', '', ''
            # lcat.append('\n')
            # gender.append('\n')
            # number.append('\n')
            # person.append('\n')
            # case.append('\n')
            # vibhakti.append('\n')
            # print(lcat, gender, number, person, case, vibhakti)
            # exit(1)
    # return lcat, gender, number, person, case, vibhakti


def write_lines_to_file(file_path, lines):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def write_generator_to_files(file_paths, generator):
    """Write contents of generator into different files."""
    lcat_file = open(file_paths[0], 'a', encoding='utf-8')
    gender_file = open(file_paths[1], 'a', encoding='utf-8')
    number_file = open(file_paths[2], 'a', encoding='utf-8')
    person_file = open(file_paths[3], 'a', encoding='utf-8')
    case_file = open(file_paths[4], 'a', encoding='utf-8')
    vibhakti_file = open(file_paths[5], 'a', encoding='utf-8')
    consolidated_file = open(file_paths[6], 'a', encoding='utf-8')
    for content in generator:
        lcat, gender, number, person, case, vibhakti, all_mor = content
        lcat_file.write(lcat + '\n')
        gender_file.write(gender + '\n')
        number_file.write(number + '\n')
        person_file.write(person + '\n')
        case_file.write(case + '\n')
        vibhakti_file.write(vibhakti + '\n')
        if all_mor.strip():
            consolidated_file.write(all_mor + '\n')
        else:
            consolidated_file.write('\n')
    lcat_file.close()
    gender_file.close()
    number_file.close()
    person_file.close()
    case_file.close()
    vibhakti_file.close()
    consolidated_file.close()


def main():
    """Pass arguments and call functions here."""
    input_morph_file = argv[1]
    domain = argv[2]
    type_of_file = argv[3]
    morph_lines = read_lines_from_file(input_morph_file)
    # lcat, gender, number, person, case, vibhakti = process_lines_and_create_labeled_data(morph_lines)
    morph_gen = process_lines_and_create_labeled_data(morph_lines)
    # print('LC', lcat, 'GEN', gender, number, person, case, vibhakti)
    lcat_file = domain + '-' + type_of_file + '-lcat.txt'
    gender_file = domain + '-' + type_of_file + '-gender.txt'
    number_file = domain + '-' + type_of_file + '-number.txt'
    person_file = domain + '-' + type_of_file + '-person.txt'
    case_file = domain + '-' + type_of_file + '-case.txt'
    vibhakti_file = domain + '-' + type_of_file + '-vibhakti.txt'
    consolidated_file = domain + '-' + type_of_file + '-all-morph.txt'
    feature_file_paths = [lcat_file, gender_file, number_file, person_file, case_file, vibhakti_file, consolidated_file]
    write_generator_to_files(feature_file_paths, morph_gen)
    # write_lines_to_file(domain + '-' + type_of_file + '-lcat.txt', lcat)
    # write_lines_to_file(domain + '-' + type_of_file + '-gender.txt', gender)
    # write_lines_to_file(domain + '-' + type_of_file + '-number.txt', number)
    # write_lines_to_file(domain + '-' + type_of_file + '-person.txt', person)
    # write_lines_to_file(domain + '-' + type_of_file + '-case.txt', case)
    # write_lines_to_file(domain + '-' + type_of_file + '-vibhakti.txt', vibhakti)


if __name__ == '__main__':
    main()
