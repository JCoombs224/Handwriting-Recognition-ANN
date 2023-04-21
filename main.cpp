#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

struct Node
{
    double collector;
    vector<Node> connections;
    vector<double> weights;
    double error;

    Node(vector<Node> connections)
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
    vector<vector<Node>> layers;
    vector<vector<double>> inputs;
    double learning_rate;

public:
    NeuralNetwork(string inputFile, string structureFile, double lrate)
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
        this->learning_rate = lrate;

        // Create layers with connections to each node based on structure
        for (int i = 0; i < structure.size(); i++)
        {
            vector<Node> layer;
            for (int j = 0; j < structure[i]; j++)
            {
                if (i == 0)
                {
                    layer.push_back(Node());
                }
                else
                {
                    layer.push_back(Node(layers[i - 1]));
                }
            }
            layers.push_back(layer);
        }

        // Load inputs from csv file into the inputs vector
        file.open(inputFile);
        if (file.is_open())
        {
            string line;
            while (getline(file, line))
            {
                vector<double> input;
                string value = "";
                for (int i = 0; i < line.length(); i++)
                {
                    if (line[i] == ',')
                    {
                        input.push_back(stod(value));
                        value = "";
                    }
                    else
                    {
                        value += line[i];
                    }
                }
                input.push_back(stod(value));
                this->inputs.push_back(input);
            }
        }
        file.close();
    }

    double sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }

    double sigmoid_derivative(double x)
    {
        return x * (1 - x);
    }

    // train the neural network using the inputs and structure with 1 output and print each epoch until the error is less than 0.05
    void train(int num_epochs, double target_error = 0.05, double l_rate = 0.1)
    {
        int num_inputs = layers[0].size();

        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
            double sum_error = 1;
            for (auto row : inputs)
            {
                vector<double> expected;
                feed_forward(row);
                for (int j = 0; j < structure[structure.size() - 1]; j++)
                {
                    expected.push_back(row[num_inputs + j]);
                    sum_error += pow(row[num_inputs + j] - layers[layers.size() - 1][j].collector, 2);
                }
                if (sum_error <= target_error)
                {
                    cout << "Target error reached error " << sum_error << endl;
                    return;
                }
                back_propagate(expected);
                update_weights(l_rate);
            }
            cout << ">Epoch=" << epoch << " l_rate=" << l_rate << " error=" << sum_error << endl;
        }
    }

    // update the weights
    void update_weights(double l_rate)
    {
        for (int i = 1; i < structure.size(); i++)
        {
            for (int j = 0; j < structure[i]; j++)
            {
                for (int k = 0; k < structure[i - 1]; k++)
                {
                    layers[i][j].weights[k] -= l_rate * layers[i][j].error * layers[i - 1][k].collector;
                }
                layers[i][j].weights[layers[i][j].weights.size()-1] -= l_rate * layers[i][j].error;
            }
        }
    }

    void forward_prop(vector<double> expected)
    {
        vector<double> error;
        for (int i = 0; i < structure[structure.size() - 1]; i++)
        {
            error.push_back(layers[structure.size() - 1][i].collector - expected[expected.size() - 1]);
        }
    }

    void feed_forward(vector<double> input)
    {
        for (int i = 0; i < structure[0]; i++)
        {
            layers[0][i].collector = input[i];
        }
        for (int i = 1; i < structure.size(); i++)
        {
            for (int j = 0; j < structure[i]; j++)
            {
                double sum = 0;
                for (int k = 0; k < structure[i - 1]; k++)
                {
                    sum += layers[i - 1][k].collector * layers[i][j].weights[k];
                }
                layers[i][j].collector = sigmoid(sum);
            }
        }
    }

    // back propagate the error
    void back_propagate(vector<double> expected)
    {
        for (int i = layers.size() - 1; i > 0; i--)
        {
            vector<Node> &layer = layers[i]; // Use reference to modify original vector
            vector<double> errors;

            if (i == layers.size() - 1) // Compare against layers.size() instead of structure.size()
            {
                for (int j = 0; j < layer.size(); j++)
                {
                    errors.push_back(layer[j].collector - expected[j]);
                }
            }
            else
            {
                for (int j = 0; j < layer.size(); j++)
                {
                    double error = 0;
                    for (auto &node : layers[i + 1]) // Use reference to modify original vector
                    {
                        error += node.weights[j] * node.error;
                    }
                    errors.push_back(error);
                }
            }

            for (int j = 0; j < layer.size(); j++)
            {
                layer[j].error = errors[j] * sigmoid_derivative(layer[j].collector);
            }

            errors.clear(); // Clear the vector for the next iteration
        }
    }

    vector<double> get_output()
    {
        vector<double> output;
        for (int i = 0; i < structure[structure.size() - 1]; i++)
        {
            output.push_back(layers[structure.size() - 1][i].collector);
        }
        return output;
    }

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
                    file << layers[i][j].weights[k] << " ";
                }
                file << endl;
            }
        }
        file.close();
    }

    // Print out the current epoch and the current error
    void print_error(int epoch, double error)
    {
        cout << "Epoch: " << epoch << " Error: " << error << endl;
    }
};

// main function
int main()
{

    NeuralNetwork nn("input.csv", "network.csv", 0.1);
    // train the neural network with the given input until the error is less than .05
    nn.train(1000, 0.05, 0.1);
    // print the output of the neural network
    vector<double> output = nn.get_output();
    for (int i = 0; i < output.size(); i++)
    {
        cout << output[i] << endl;
    }
    return 0;
}
