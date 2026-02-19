train_file=$1
output_folder=$2
lang=$3
type_of_file=$4
if [ ! -d $output_folder ];then
	mkdir $output_folder
fi
cut -f1-2 $train_file > $output_folder"/"$lang"-"$type_of_file"-token-pos.txt"
cut -f1-2,8 $train_file > $output_folder"/"$lang"-"$type_of_file"-token-pos-chunk.txt"
cut -f1,3 $train_file > $output_folder"/"$lang"-"$type_of_file"-token-lcat.txt"
cut -f1,4 $train_file > $output_folder"/"$lang"-"$type_of_file"-token-gender.txt"
cut -f1,5 $train_file > $output_folder"/"$lang"-"$type_of_file"-token-number.txt"
cut -f1,6 $train_file > $output_folder"/"$lang"-"$type_of_file"-token-person.txt"
cut -f1,7 $train_file > $output_folder"/"$lang"-"$type_of_file"-token-case.txt"
