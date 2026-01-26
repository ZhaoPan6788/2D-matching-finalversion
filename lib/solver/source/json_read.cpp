/*-----------------------------------------------------------------------------
Json class
Simply encapsulate the JSON object and be able to read the JSON file

@time       :   2021/12/24
@author     :   zili.chen
@version    :   v1.0
@log        :   v1.0 first build
-----------------------------------------------------------------------------*/

// include
#include "json_read.hpp"


JsonRead::JsonRead(string filePath)
{
    this->read(filePath);
}

JsonRead::~JsonRead()
{
    
}


/*-----------------------------------------------------------------------------
Function read
from filePath read Json object
    filePath    json file's path

-----------------------------------------------------------------------------*/
void JsonRead::read(string filePath)
{
    this->filePath = filePath;

    // open
    this->fin.open(this->filePath, ios::in);
    if (!this->fin.is_open())
        std::cout << "Read transLine config file error, not open this file!" << endl;

    // read string
    this->buffer << this->fin.rdbuf();

    // string to JSON object
    this->data = json::parse(this->buffer.str());

    // print string
    // std::cout << endl << "load Config: " << endl;
    // std::cout << this->data.dump(4) << endl << endl;

    fin.close();
}