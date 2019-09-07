package com.zzk;
import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.URL;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
public class PartImageFrame extends JFrame {
    private Image img = null;  // 声明图像对象
    private PartImagePanel imagePanel = null;  // 声明图像面板对象
    private int dx1, dy1, dx2, dy2;   // 目标矩形第一个角与第二个角的X、Y坐标
    private int sx1, sy1, sx2, sy2;   // 源矩形第一个角与第二个角的X、Y坐标
    public static void main(String args[]) {
        PartImageFrame frame = new PartImageFrame();
        frame.setVisible(true);
    }
    public PartImageFrame() {
        super();
        URL imgUrl = PartImageFrame.class.getResource("/img/image.jpg");// 获取图片资源的路径
        img = Toolkit.getDefaultToolkit().getImage(imgUrl); // 获取图像资源
        dx2 = sx2 = 340; // 初始化图像大小
        dy2 = sy2 = 200; // 初始化图像大小
        imagePanel = new PartImagePanel();  // 创建图像面板对象
        this.setBounds(200, 160, 355, 276); // 设置窗体大小和位置
        this.add(imagePanel); // 在窗体中部位置添加图像面板对象
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // 设置窗体关闭模式
        this.setTitle("翻转图像"); // 设置窗体标题
        final JPanel panel = new JPanel();
        getContentPane().add(panel, BorderLayout.SOUTH);
        final JButton btn_h = new JButton();
        btn_h.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                // 下面3行代码用于交换sx1和sx2的值
                int x = sx1;
                sx1 = sx2;
                sx2 = x;
                imagePanel.repaint();  // 重新调用面板类的paint()方法
            }
        });
        btn_h.setText("水平翻转");
        panel.add(btn_h);
        final JButton btn_v = new JButton();
        btn_v.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                // 下面3行代码用于交换sy1和sy2的值
                int y = sy1;
                sy1 = sy2;
                sy2 = y;
                imagePanel.repaint();// 重新调用面板类的paint()方法
            }
        });
        btn_v.setText("垂直翻转");
        panel.add(btn_v);
    }
    // 创建面板类
    class PartImagePanel extends JPanel {
        public void paint(Graphics g) {
            g.clearRect(0, 0, this.getWidth(), this.getHeight());// 清除绘图上下文的内容
            g.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, this);// 绘制图像
        }
    }
}
