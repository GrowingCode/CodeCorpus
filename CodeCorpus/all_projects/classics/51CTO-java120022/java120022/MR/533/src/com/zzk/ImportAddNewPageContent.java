package com.zzk;

import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Element;
import com.itextpdf.text.PageSize;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;

public class ImportAddNewPageContent {
	public static void main(String[] args) {
		try {
			PdfReader reader = new PdfReader("c:\\创建第一个PDF文档.pdf");// 导入文档
			PdfStamper stamp = new PdfStamper(reader, new FileOutputStream(
					"c:\\导入并添加新页和内容.pdf"));// 关联文档与输出流
			BaseFont Chinese = BaseFont.createFont("STSong-Light",
					"UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
			PdfContentByte cb = stamp.getOverContent(1);// 获取第一页内容
			cb.beginText();// 写内容
			cb.setFontAndSize(Chinese, 25);// 设置字体属性
			cb.setTextMatrix(15, 15);// 设置矩阵(坐标)
			cb.showText("第一页");// 矩阵处显示文本
			cb.showTextAligned(Element.ALIGN_CENTER, "新增的内容。", 180, 760, 0);// 设置文本对齐，内容，位置和旋转角度
			cb.endText();// 内容结束
			stamp.insertPage(2, PageSize.A4);// 增加新页
			cb = stamp.getOverContent(2);// 获取第2页内容
			cb.beginText();// 写内容
			cb.setFontAndSize(Chinese, 20);// 设置字体属性
			cb.showTextAligned(Element.ALIGN_LEFT, "在新增的页中添加的内容。", 100, 600, 0);// 设置文本对齐，内容，位置和旋转角度
			cb.endText();// 内容结束
			stamp.close();// 关闭
		} catch (IOException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
