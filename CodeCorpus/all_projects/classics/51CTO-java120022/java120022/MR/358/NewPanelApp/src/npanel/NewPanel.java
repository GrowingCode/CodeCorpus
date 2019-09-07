package npanel;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JTextField;
public class NewPanel extends JPanel {
    private JTextField textField;// 声明文本框
    public NewPanel() {
        super();// 调用超类的构造方法
        setLayout(null);// 设置面板为绝对布局
        setBounds(100, 100, 254, 167);// 设置面板的位置和大小
        final JButton button = new JButton();// 创建按钮
        button.setText("按钮一");// 指定按钮标题
        button.setBounds(38, 86, 73, 28);// 指定按钮在面板中的显示位置和大小
        add(button);// 将按钮添加到面板上
        final JButton button_1 = new JButton();// 创建按钮
        button_1.setText("按钮二");// 指定按钮标题
        button_1.setBounds(140, 86, 73, 28);// 指定按钮在面板中的显示位置和大小
        add(button_1);// 将按钮添加到面板上
        textField = new JTextField();// 创建文本框
        textField.setText("这是一个文本框控件");// 指定文本框的标题
        textField.setBounds(38, 35, 175, 22);// 设置文本框的显示位置和大小
        add(textField);// 将文本框添加到面板上
    }
}
