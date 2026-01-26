# 删除数据文件
rm *.dat
rm *.out
rm *.log
rm *.h5
rm *.txt

# -f 参数判断 $file 是否存在
if [ ! -d "./restart" ];then
  mkdir restart
else
  rm -rf restart
  mkdir restart
fi

if [ ! -d "./diag" ];then
  mkdir diag
else
  rm -rf diag
  mkdir diag
fi

if [ ! -d "./check" ];then
  mkdir check
else
  rm -rf check
  mkdir check
fi
