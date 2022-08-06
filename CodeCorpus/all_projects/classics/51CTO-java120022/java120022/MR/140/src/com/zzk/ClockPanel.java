package com.zzk;

import static java.util.Calendar.DAY_OF_MONTH;
import static java.util.Calendar.DAY_OF_WEEK;
import static java.util.Calendar.HOUR;
import static java.util.Calendar.MILLISECOND;
import static java.util.Calendar.MINUTE;
import static java.util.Calendar.MONTH;
import static java.util.Calendar.SECOND;
import static java.util.Calendar.YEAR;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.font.FontRenderContext;
import java.awt.geom.Rectangle2D;
import java.util.Calendar;
import javax.swing.ImageIcon;
import javax.swing.JPanel;

/**
 * 时钟面板
 * 
 * @author 张振坤
 */
@SuppressWarnings("serial")
public class ClockPanel extends JPanel {
    // 3个指针的粗细
    private static final BasicStroke HOURS_POINT_WIDTH = new BasicStroke(4);
    private static final BasicStroke MINUETES_POINT_WIDTH = new BasicStroke(3);
    private static final BasicStroke SEC_POINT_WIDTH = new BasicStroke(2);
    ImageIcon background;// 背景图片对象
    private int centerX;// 中心坐标
    private int centerY;// 中心坐标
    private final static int secLen = 60; // 指针长度
    private final static int minuesLen = 55; // 指针长度
    private final static int hoursLen = 36; // 指针长度
    
    /**
     * 构造方法
     */
    public ClockPanel() {
        setToolTipText("小键盘+、-调整透明度，Crtl+Shift+X退出");// 提示信息
        setOpaque(false);
        
        background = new ImageIcon(getClass().getResource("clock.jpg"));// 加载图片
        int iconWidth = background.getIconWidth();// 获取图片大小
        centerX = iconWidth / 2;// 获取图片中间坐标
        int iconHeight = background.getIconHeight();
        centerY = iconHeight / 2;// 获取图片中间坐标
        setPreferredSize(new Dimension(iconWidth, iconHeight));
    }
    
    @Override
    public void paint(Graphics g) {// 重写父类方法
        Graphics2D g2 = (Graphics2D) g;
        g2.drawImage(background.getImage(), 0, 0, this);// 绘制背景图片
        Calendar calendar = Calendar.getInstance();// 获取日历对象
        int millisecond = calendar.get(MILLISECOND);// 获取毫秒值
        int sec = calendar.get(SECOND);// 获取秒值
        int minutes = calendar.get(MINUTE);// 获取分值
        int hours = calendar.get(HOUR);// 获取时值
        String dateStr = calendar.get(YEAR) + "年" + (calendar.get(MONTH) + 1)
                + "月" + calendar.get(DAY_OF_MONTH) + "日";// 当前日期字符串
        int weekDay = calendar.get(DAY_OF_WEEK) - 1;// 当前星期值
        String weekStr = "";
        if (weekDay == 0) {
            weekStr = "星期日";
        } else {
            weekStr = "星期" + weekDay;
        }
        
        {// 绘制日期与星期
            g2.setColor(Color.BLACK);// 设置背景色
            g2.setFont(new Font("楷体", Font.BOLD, 12));// 设置字体
            // 获取字体渲染上下文
            FontRenderContext fontRenderContext = g2.getFontRenderContext();
            // 计算日期字符串的边界大小
            Rectangle2D dateStrBounds = getFont().getStringBounds(dateStr,
                    fontRenderContext);
            // 在时钟下半部绘制日期
            g2.drawString(dateStr,
                    (int) (centerX - dateStrBounds.getWidth() / 2),
                    centerY + 20);
            // 改变字体大小
            g2.setFont(new Font("楷体", Font.BOLD, 14));
            // 计算星期字符串边界大小
            Rectangle2D weekStrBounds = getFont().getStringBounds(weekStr,
                    fontRenderContext);
            // 绘制星期字符串
            g2.drawString(weekStr,
                    (int) (centerX - weekStrBounds.getWidth() / 2),
                    centerY + 35);
        }
        double secAngle = (60 - sec) * 6 - (millisecond / 150); // 秒针角度
        int minutesAngle = (60 - minutes) * 6;// 分针角度
        int hoursAngle = (12 - hours) * 360 / 12 - (minutes / 2);// 时针角度
        // 计算秒针、分针、时针指向的坐标
        int secX = (int) (secLen * Math.sin(Math.toRadians(secAngle)));// 秒针指向点的X坐标
        int secY = (int) (secLen * Math.cos(Math.toRadians(secAngle))); // 秒针指向点的Y坐标
        int minutesX = (int) (minuesLen * Math
                .sin(Math.toRadians(minutesAngle))); // 分针指向点的X坐标
        int minutesY = (int) (minuesLen * Math
                .cos(Math.toRadians(minutesAngle))); // 分针指向点的Y坐标
        int hoursX = (int) (hoursLen * Math.sin(Math.toRadians(hoursAngle))); // 时针指向点的X坐标
        int hoursY = (int) (hoursLen * Math.cos(Math.toRadians(hoursAngle))); // 时针指向点的Y坐标
        // 分别绘制时针、分针、秒针
        g2.setStroke(HOURS_POINT_WIDTH);// 设置时针的宽度
        g2.setColor(Color.BLACK);// 设置时针的颜色
        g2.drawLine(centerX, centerY, centerX - hoursX, centerY - hoursY);// 绘制时针
        g2.setStroke(MINUETES_POINT_WIDTH);// 设置分针的宽度
        if (minutesAngle != hoursAngle) // 分针、时针重叠变色
            g2.setColor(new Color(0x2F2F2F));// 设置未重叠时的颜色
        else {
            g2.setColor(Color.GREEN);// 设置重叠时的颜色
        }
        g2.drawLine(centerX, centerY, centerX - minutesX, centerY - minutesY);// 绘制分针
        g2.setStroke(SEC_POINT_WIDTH);// 设置秒针的宽度
        if (secAngle != hoursAngle && secAngle != minutesAngle)// 分针、时针、秒针重叠变色
            g2.setColor(Color.ORANGE);// 设置未重叠时的颜色
        else {
            g2.setColor(Color.GREEN);// 设置重叠时的颜色
        }
        // 绘制3个指针的中心圆和秒针
        g2.fillOval(centerX - 5, centerY - 5, 10, 10);// 绘制中心圆
        g2.drawLine(centerX, centerY, centerX - secX, centerY - secY);// 绘制秒针
        g2.drawLine(centerX + 1, centerY + 1, centerX - secX + 1, centerY
                - secY + 1);// 绘制秒针
    }
}
