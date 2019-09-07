package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.*;


public class HandleChineseErrorCode {
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("C:\\亚运速递.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			BaseFont Chinese = BaseFont.createFont("STSong-Light", "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体   
			Font FontChinese = new Font(Chinese, 20, Font.NORMAL);// 实例化字体类与设置字体大小属性           
			document.add(new Paragraph("中国再一次实现了金牌数和奖牌数第一的目标", FontChinese));// 向文档中添加内容并指定中文
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