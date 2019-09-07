package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class SetParagraphSpacingAfter {
	public static void main(String[] args){
		Document document = new Document();		//创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\Java.pdf"));// 关联文档对象与输出流
			document.open();					//打开文档
			Paragraph paragraph1 = new Paragraph("Java resource");	//创建段落添加内容
			paragraph1.setSpacingBefore(10); 	//设置段落上边距
			paragraph1.setSpacingAfter(30); 	//设置段落下边距
			document.add(paragraph1);			//向文档添加段落
			Paragraph paragraph2 = new Paragraph("Java classes introduce");			//创建段落添加内容
			paragraph2.setSpacingAfter(30); 	//设置段落下边距
			document.add(paragraph2);			//向文档添加段落
			document.close();					//关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
