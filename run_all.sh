FOLDERS=new_all_data/*.*
for f in $FOLDERS
do
	echo "Processing $f"
	python run_on_dataset.py $f 5000 5000
done