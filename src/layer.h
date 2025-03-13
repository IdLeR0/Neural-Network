#include "activation_function.h"

#include "linalg.h"

namespace network{

    class Layer{
        public:
        
        void Linear(int input_size, int output_size);
        void SetActivationFuntion(NamesActivationFunc name);
        Vector Forward(const Vector& input);
        Matrix Forward(const Matrix& input);
        Vector Backward(const Vector& gradient);
        Matrix Backward(const Matrix& gradient);

        

        private:
        ActivationFunc func_;
        Matrix weigts_;
        Vector bias_;
        bool is_set_func_;
        void SetRandomWeigts();
    };
   
}