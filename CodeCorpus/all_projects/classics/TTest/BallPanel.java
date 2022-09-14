package TTest;
import java.awt.*;
import javax.swing.JLabel;
/**
 * @author 张振坤
 */
public class BallPanel extends JLabel implements Runnable {
    private int r = 10;// 小球半径
    private int width = r * 2;// 球宽度
    private int height = r * 2;// 球高度
    private Color ballColor = Color.BLUE;// 默认颜色
    public BallPanel() {
        setPreferredSize(new Dimension(width, height));// 初始化大小
        new Thread(this).start();// 启动小球跳跃线程
    }
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(ballColor);// 设置默认颜色
        g.fillOval(0, 0, width, height);// 在标签上绘制球体
    }
    @Override
    public void run() {
        Container parent = getParent();/// 获得当前标签的父级容器对象
        Point myPoint = getLocation();// 获取初始位置
        while (true) {// 循环读取父容器对象
            if (parent == null) {
                try {
                    Thread.sleep(50);// 线程休眠
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                myPoint = getLocation();// 获取初始位置
                parent = getParent();// 获得当前标签的父级容器对象
            } else {// 如果已经获取到父容器
                break;// 跳出循环
            }
        }
        int sx = myPoint.x;// X坐标
        int sy = myPoint.y;// y坐标
        int step = (int) (Math.random() * 10) % 3 + 1;// 移动步进
        int dx = (Math.random() * 100) >= 50 ? step : -step;// 水平步进值
        step = (int) (Math.random() * 10) % 3 + 1;// 移动步进
        int dy = (Math.random() * 100) >= 50 ? step : -step;// 垂直步进值
        int stime = (int) (Math.random() * 80 + 10);// 随机移动速度
        while (parent.isVisible()) {
            int parentWidth = parent.getWidth();// 容器宽度
            int parentHeight = parent.getHeight();// 容器高度
            setLocation(sx, sy);// 指定小球的位置
            try {
                Thread.sleep(stime);// 通过休眠改变移动速度
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            sx = sx + dx * 5;// 水平坐标偏移5个像素
            sy = sy + dy * 5;// 垂直坐标偏移5个像素
            // 检测垂直边界
            if (sy > parentHeight - height || sy < 0)
                dy = -dy;// 防止坐标超出垂直边界
            // 检测水平边界
            if (sx > parentWidth - width || sx < 0)
                dx = -dx;// 防止坐标超出水平边界
        }
    }
}
