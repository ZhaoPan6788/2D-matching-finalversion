#ifndef _JSON_READ_
#define _JSON_READ_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <json.hpp>

using json = nlohmann::json;
using namespace std;

class JsonRead
{
public:
    JsonRead(string filePath);

    JsonRead(const JsonRead &src) = delete;

    JsonRead &operator=(const JsonRead &) = delete;

    ~JsonRead();

    void read(string filePath);

    json                    data;

private : 
    ifstream                fin;
    ofstream                fout;
    string                  filePath;
    stringstream            buffer;
};


#endif