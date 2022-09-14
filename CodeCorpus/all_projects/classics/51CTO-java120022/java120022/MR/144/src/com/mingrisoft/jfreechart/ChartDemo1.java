/**
 * @jdk版本:1.6
 * @编码时间:2010-7-20
 */
package com.mingrisoft.jfreechart;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.general.PieDataset;

public class ChartDemo1 {

	/**
	 * 生成JFreeChart
	 * 
	 * @return
	 */
	public JFreeChart getJFreeChart() {
		PieDataset dataset = getPieDataset();
		JFreeChart chart = ChartFactory.createPieChart("Pie title",
				dataset, false, false, false);
		return chart;
	}

	/**
	 * 创建一个饼图表的数据集
	 * 
	 * @return
	 */
	private PieDataset getPieDataset() {
		DefaultPieDataset dataset = new DefaultPieDataset();
		dataset.setValue("A", 200);
		dataset.setValue("B", 400);
		dataset.setValue("C", 500);
		return dataset;
	}

	public static void main(String[] args) {
		ChartDemo1 chartDemo1 = new ChartDemo1();
		ChartFrame chartFrame = new ChartFrame("JFreeChar Demo",
				chartDemo1.getJFreeChart());
		chartFrame.pack();
		chartFrame.setVisible(true);

	}
}
