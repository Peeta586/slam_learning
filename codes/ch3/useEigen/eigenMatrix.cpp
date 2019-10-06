#include <iostream>

using namespace std;

#include <ctime>

// 核心
#include <Eigen/Core>
// 稠密矩阵的代数运算 （逆， 特征值）
#include <Eigen/Dense> // 包括了Corem, Geometry, LU, Cholesky, SVD, QR and Eigenvalues

using namespace Eigen;


#define MATRIX_SIZE 50

int main(int argc, char ** argv){
    // Eigen 中所有向量和矩阵都是Eigen::Matrix，它是一个模板类。它的前三个参数为：数据类型，行，列
    // 声明一个2*3的float矩阵
    // 该类的模板形式可能如：template<class T,int H, int W>
    // Matrix<int,3, 4, ColMajor> Acolmajor; Matrix创建的矩阵默认是按列存储
    Matrix<float, 2, 3> matrix_23;

    // 另一种定义形式
    // 同时，Eigen 通过 typedef 提供了许多内置类型，不过底层仍是Eigen::Matrix
    // 例如 Vector3d 实质上是 Eigen::Matrix<double, 3, 1>，即三维向量
    Vector3d v_3d;  // Matrix<double, 3, 1> d表示double

    Matrix<double, 3,1> vd_3d;

    // Matrix3d 实质上是 Eigen::Matrix<double, 3, 3>
    // 注意这个声明一般都是3,4所有的，没有像Matrix10f这样的定义
    Matrix3d matrix_33 = Matrix3d::Zero(); //初始化为零
    Matrix3f matrix_1010 = Matrix3f::Zero();
    // 也就是如果Matrix<float, 2, 3>写嫌麻烦，那么可以用包装好的Vector3d，Matrix3d 去声明变量


    /*************************************动态矩阵 */
    // 动态矩阵和静态矩阵：动态矩阵是指其大小在运行时确定，静态矩阵是指其大小在编译时确定。
    // 如果不确定矩阵大小，可以使用动态大小的矩阵
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;
    // 更简单的
    MatrixXd matrix_x;

    /*************** 具体操作 */
    // 下面是对Eigen阵的操作
    // 输入数据（初始化）
    matrix_23 << 1, 4, 3, 4, 9, 6;
    // 输出
    cout<< "matrix 2x3 from random: \n" << matrix_23<< endl;
    cout<< "(0,0)--->" << matrix_23(0,0)<<endl;

    // 用()访问矩阵中的元素
    cout << "print matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
        cout << endl;
    }
    
    // 矩阵和向量相乘（实际上仍是矩阵和矩阵）
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    // 但是在Eigen里你不能混合两种不同类型的矩阵，像这样是错的
    // Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
    // 应该显式转换
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

    // Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    // cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl;

    // 同样你不能搞错矩阵的维度
    // 试着取消下面的注释，看看Eigen会报什么错
    // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

    // 一些矩阵运算
    // 四则运算就不演示了，直接用+-*/即可。
    matrix_33 = Matrix3d::Random();  // 随机数矩阵
    cout << "random matrix \n" << matrix_33 <<endl;
    cout << "transpose \n" << matrix_33.transpose() << endl;
    cout << "sum: \n" << matrix_33.sum() << endl;
    cout << "trace: \n" << matrix_33.trace() << endl;
    cout << "times 10: \n" << 10 * matrix_33 << endl;
    cout << "inverse: \n" << matrix_33.inverse() << endl;
    cout << "det(行列式): " << matrix_33.determinant() << endl;

    // 特征值
    // 实对称矩阵可以保证对角化成功
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values: \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors: \n" << eigen_solver.eigenvectors() << endl;

    // 解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程
    // N的大小在前边的宏里定义，它由随机数生成
    // 直接求逆自然是最直接的，但是求逆运算量大
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
        = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE); // 记住声明方式
    matrix_NN = matrix_NN * matrix_NN.transpose(); // 保证半正定
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();

    // 直接求逆
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inverse is" <<
        1000*(clock() - time_stt) / (double) CLOCKS_PER_SEC <<"ms" <<endl;
    cout << "x = " << x.transpose() << endl;

    // 通常用矩阵分解來求， 例如QR分解速度会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decomposition is" <<
        1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC <<"ms"<<endl;
    cout << "x = " << x.transpose() << endl; 

    // 对于正定矩阵，还可以用cholesky分解来解方程
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    

    return 0;
}

//  Map类：在已经存在的矩阵或向量中，不必拷贝对象，而是直接在该对象的内存上进行运算操作。
template <typename T>  
static void matrix_mul_matrix(T* p1, int iRow1, int iCol1, T* p2, int iRow2, int iCol2, T* p3)  
{  
    if (iRow1 != iRow2) return;  
  
    //列优先  
    //Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map1(p1, iRow1, iCol1);  
    //Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map2(p2, iRow2, iCol2);  
    //Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map3(p3, iCol1, iCol2);  
  
    //行优先  
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map1(p1, iRow1, iCol1);  
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map2(p2, iRow2, iCol2);  
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map3(p3, iCol1, iCol2);  
  
    map3 = map1 * map2;  
}  