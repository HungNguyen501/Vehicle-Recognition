- random_state: 
Nếu bạn không chỉ định random_statemã trong mã của mình thì mỗi lần bạn chạy 
(thực thi) mã của mình sẽ tạo ra một giá trị ngẫu nhiên mới và các bộ dữ liệu 
kiểm tra và huấn luyện sẽ có các giá trị khác nhau mỗi lần.

Tuy nhiên, nếu một giá trị cố định được gán như thế random_state = 42 thì cho dù 
bạn có thực thi mã của mình bao nhiêu lần thì kết quả sẽ giống nhau .ie, cùng các 
giá trị trong tập dữ liệu thử nghiệm và kiểm tra.

- C:
C is a regularization parameter that controls the trade off between 
the achieving a low training error and a low testing error that is the 
ability to generalize your classifier to unseen data. Consider the objective 
function of a linear SVM








