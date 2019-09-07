package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.Chunk;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfWriter;

public class SetSubSupScrip {
	public static void main(String[] args) {
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream(
					"C:\\设置上标和下标.pdf"));// 关联文档对象与输出流
			document.open();
			BaseFont Chinese = BaseFont.createFont("STSong-Light",
					"UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
			Font fontChinese = new Font(Chinese, 20, Font.NORMAL);// 实例化字体类与设置字体大小属性
			document.add(new Paragraph("下面是使用上标的效果：", fontChinese));// 添加段落
			Chunk chunk = new Chunk("X");// 创建块
			document.add(chunk);// 向文档添加内容
			chunk = new Chunk("2");// 创建块
			chunk.setTextRise(4.0f);// 提升块文本
			document.add(chunk);// 添加块
			chunk = new Chunk("+Y");// 创建块
			document.add(chunk);// 添加块
			chunk = new Chunk("2");// 创建块
			chunk.setTextRise(4.0f);// 提升块文本
			document.add(chunk);// 添加块
			document.add(new Paragraph("下面是使用下标的效果：", fontChinese));// 添加段落
			chunk = new Chunk("M");// 创建块
			document.add(chunk);// 添加块
			chunk = new Chunk("2");// 创建块
			chunk.setTextRise(-3.0f);// 降低块文本
			document.add(chunk);// 添加块
			chunk = new Chunk("+N");// 创建块
			document.add(chunk);// 添加块
			chunk = new Chunk("2");// 创建块
			chunk.setTextRise(-3.0f);// 降低块文本
			document.add(chunk);// 添加块
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
