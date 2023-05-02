#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h> 

using namespace std;

// SQLite3 database variables
sqlite3 *db;
char *zErrMsg = 0;
int rc;
char *sql;

struct Node
{
    double collector;
    vector<Node *> connections;
    vector<double> weights;
    double error;

    Node(vector<Node *> connections)
    {
        this->connections = connections;
        this->collector = 0;
        for (int i = 0; i < connections.size(); i++)
        {
            weights.push_back((double)rand() / RAND_MAX);
        }
    }

    Node()
    {
        this->collector = 0;
    }
};

class NeuralNetwork
{
private:
    vector<int> structure;
    vector<vector<Node *>> network;
    vector<vector<double>> inputs;

    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double collector)
    {
        return collector * (1.0 - collector);
    }

    // update the weights
    void update_weights(double l_rate)
    {
        for (int i = 1; i < network.size(); i++)
        {
            vector<double> inputs;
            for (auto neuron : network[i - 1])
            {
                inputs.push_back(neuron->collector);
            }
            for (auto neuron : network[i])
            {
                for (int j = 0; j < inputs.size(); j++)
                {
                    neuron->weights[j] -= l_rate * neuron->error * inputs[j];
                }
                // neuron->weights.back() -= l_rate * neuron->error;
            }
        }
    }

    // feed forward the inputs
    void feed_forward(vector<double> row)
    {
        for (int i = 0; i < structure[0]; i++)
        {
            network[0][i]->collector = row[i];
        }
        for (int i = 1; i < structure.size(); i++)
        {
            for (int j = 0; j < structure[i]; j++)
            {
                double sum = 0;
                for (int k = 0; k < structure[i - 1]; k++)
                {
                    sum += network[i - 1][k]->collector * network[i][j]->weights[k];
                }
                network[i][j]->collector = sigmoid(sum);
            }
        }
    }

    // back propagate the error
    void back_propagate(vector<double> expected)
    {
        for (int i = network.size() - 1; i > 0; i--)
        {
            vector<Node *> layer = network[i];
            vector<double> errors;

            if (i == network.size() - 1)
            {
                for (int j = 0; j < layer.size(); j++)
                {
                    errors.push_back(layer[j]->collector - expected[j]);
                }
            }
            else
            {
                for (int j = 0; j < layer.size(); j++)
                {
                    double error = 0.0;
                    for (auto node : network[i + 1])
                    {
                        error += node->weights[j] * node->error;
                    }
                    errors.push_back(error);
                }
            }
            // 
            for (int j = 0; j < layer.size(); j++)
            {
                layer[j]->error = errors[j] * sigmoid_derivative(layer[j]->collector);
            }

            errors.clear(); // Clear the vector for the next iteration
        }
    }
public:
    NeuralNetwork(string structureFile)
    {
        // Load structure from file into the structure vector
        ifstream file(structureFile);
        if (file.is_open())
        {
            char c;
            string val;
            while (file >> c)
            {
                if (c == ',')
                {
                    structure.push_back(stoi(val));
                    val = "";
                }
                else
                {
                    val += c;
                }
            }
            structure.push_back(stoi(val));
        }
        file.close();

        // Create network with connections to each node based on structure
        for (int i = 0; i < structure.size(); i++)
        {
            vector<Node *> layer;
            for (int j = 0; j < structure[i]; j++)
            {
                if (i == 0)
                {
                    layer.push_back(new Node());
                }
                else
                {
                    layer.push_back(new Node(network[i - 1]));
                }
            }
            network.push_back(layer);
        }
    }

    // train the neural network
    void train(const char* query, int num_epochs, double target_error = 0.05, double l_rate = 0.1)
    {
        int num_inputs = network[0].size();
        inputs.clear();

        // get the inputs from the database
        sqlite3_stmt *stmt;
        rc = sqlite3_prepare_v2(db, query, -1, &stmt, NULL);
        if (rc != SQLITE_OK)
        {
            fprintf(stderr, "SQL error: %s\n", zErrMsg);
            sqlite3_free(zErrMsg);
            return;
        }

        // push the inputs into the inputs vector
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            vector<double> input;
            for (int i = 0; i < num_inputs; i++)
            {
                input.push_back(sqlite3_column_double(stmt, i));
            }
            for (int i = num_inputs; i < sqlite3_column_count(stmt); i++)
            {
                input.push_back(sqlite3_column_double(stmt, i));
            }
            this->inputs.push_back(input);
        }
        sqlite3_finalize(stmt);
        sqlite3_close(db);

        cout << "Number of inputs = " << inputs.size() << endl;
        // train the network
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            double epoch_error = 0;
            for (auto row : inputs)
            {
                vector<double> expected;
                feed_forward(row);
                for (int j = 0; j < structure[structure.size() - 1]; j++)
                {
                    expected.push_back(row[num_inputs + j]);
                }
                for (int j = 0; j < structure[structure.size() - 1]; j++)
                {
                    epoch_error += pow(network[structure.size() - 1][j]->collector - expected[j], 2);
                }
                back_propagate(expected);
                update_weights(l_rate);
            }
            if (epoch_error <= target_error)
            {
                cout << "Target error reached error " << epoch_error << endl;
                return;
            }
            cout << setprecision(10) << ">Epoch=" << epoch << " l_rate=" << l_rate << " error=" << setprecision(20) << epoch_error << endl;
        }
    }

    // run with a vector of inputs and print output
    void run(vector<double> inputs)
    {
        feed_forward(inputs);
        for (int i = 0; i < structure[structure.size() - 1]; i++)
        {
            cout << network[structure.size() - 1][i]->collector << " ";
        }
        cout << endl;
    }


    vector<double> get_output()
    {
        vector<double> output;
        for (int i = 0; i < structure[structure.size() - 1]; i++)
        {
            output.push_back(network[structure.size() - 1][i]->collector);
        }
        return output;
    }

    // save the weights to a file
    void save(string filename)
    {
        ofstream file;
        file.open(filename);
        for (int i = 1; i < structure.size(); i++)
        {
            for (int j = 0; j < structure[i]; j++)
            {
                for (int k = 0; k < structure[i - 1]; k++)
                {
                    file << network[i][j]->weights[k] << " ";
                }
                file << endl;
            }
        }
        file.close();
    }
};



// main function
int main()
{
    // seed random number generator
    srand(time(NULL));

    // Open database FOR UNIX REQUIRES: sudo apt install libsqlite3-dev
    rc = sqlite3_open("hw_data_2", &db);

    // Check if database opened correctly
    if(rc)
    {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return -1;
    }
    else
    {
        fprintf(stderr, "Opened database successfully\n");
    }
    
    // create the neural network
    NeuralNetwork nn("network.csv");

    // train the neural network with the given input until the error is less than .10

    // training for letter A
    // cout << "Training for letter a" << endl;
    // nn.train("select * from a_train order by random() limit 1000;", 10000, 0.10, 0.1);

    // training for letter B
    // cout << "Training for letter b" << endl;
    // nn.train("select * from b_train order by random();", 100, 0.10, 0.1);

    // training for letter C
    // cout << "Training for letter c" << endl;
    // nn.train("select * from c_train order by random() limit 3000;", 10000, 0.10, 0.01);

    // training for letter D
    // cout << "Training for letter d" << endl;
    // nn.train("select * from d_train order by random() limit 5000;", 10000, 0.10, 0.6);

    // training for letter E
    cout << "Training for letter e" << endl;
    nn.train("select * from e_train order by random() limit 500;", 10000, 0.10, 0.5);

    // print the output of the neural network
    vector<double> output = nn.get_output();
    for (int i = 0; i < output.size(); i++)
    {
        cout << output[i] << endl;
    }

    return 0;
}
