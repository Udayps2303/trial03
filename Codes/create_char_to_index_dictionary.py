"""Create character to index dictionary and save them as a pickle."""
from sys import argv
from pickle import dump


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def create_character_set_from_lines(lines):
    """Create a character set from lines."""
    characters = set()
    for line in lines:
        token = line.split('\t')[0]
        characters.update(set(token))
    return characters


def create_char_to_index_dict(characters):
    """Create a char2index dictionary from a set of characters."""
    return {char: index + 1 for index, char in enumerate(characters)}


def dump_object_into_pickle_file(data_object, pickle_file):
    """Dump an object into a pickle file."""
    with open(pickle_file, 'wb') as pickle_dump:
        dump(data_object, pickle_dump)


def main():
    """Pass arguments and call functions here."""
    input_file = argv[1]
    pickle_file = argv[2]
    input_lines = read_lines_from_file(input_file)
    input_characters = create_character_set_from_lines(input_lines)
    char2index_dict = create_char_to_index_dict(input_characters)
    # print(char2index_dict)
    dump_object_into_pickle_file(char2index_dict, pickle_file)


if __name__ == '__main__':
    main()      