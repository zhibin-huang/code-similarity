# installation
cd ASTConverter
mvn clean package
cd ..

# compile cpp module
./Core.Pipeline/native/compile.sh ./Core.Pipeline/native/cpp_module

# compile entire corpus to ast json file and create jsrc.json from all Java files under jsrc
time mvn exec:java -Dexec.mainClass=ConvertJava -Dexec.args="compilationUnit /Users/huangzhibin/Desktop/jsrc.json /Users/huangzhibin/Desktop/java1000"

# convert query_file.java into ast in json format
time mvn exec:java -Dexec.mainClass=ConvertJava -Dexec.args="compilationUnit Example/example_query.json Example/example_query.java"

# featurize corpus
time python3 Service/entry.py -c ./Repos -d ./Obj

# run experiments assuming that featurization has already been done
time python3 -m cProfile -o Log/profiler Service/entry.py -d ./Obj -t

# search code at index 83403 of the corpus
time python3 Service/entry.py -d ./Obj -i 83403

# search using the query ast in query_file.json
time python3 Service/entry.py -d ./Obj -f query_file.json

# read profile
python3 read_profiler.py

# start service
python3 Service/entry.py -d ./Obj
